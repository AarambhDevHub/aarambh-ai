use std::cmp::Ordering;
use std::collections::HashMap;

use aarambh_ai_core::DsaConfig;
use candle_core::{DType, Module, Result, Tensor};
use candle_nn::Linear;

use crate::attention::GroupedQueryAttention;
use crate::kvcache::DsaKvCache;
use crate::rope::RopeCache;

#[derive(Debug, Clone, Default)]
/// Runtime counters emitted by learned block-sparse attention.
pub struct DsaForwardStats {
    /// Number of query rows evaluated by DSA layers.
    pub queries: usize,
    /// Total selected causal blocks across query rows.
    pub selected_blocks: usize,
    /// Total selected K/V token rows across query rows.
    pub selected_tokens: usize,
    /// Number of forwards that used exact dense attention.
    pub dense_fallbacks: usize,
}

impl DsaForwardStats {
    /// Merge counters from another layer or forward.
    pub fn merge(&mut self, other: &Self) {
        self.queries += other.queries;
        self.selected_blocks += other.selected_blocks;
        self.selected_tokens += other.selected_tokens;
        self.dense_fallbacks += other.dense_fallbacks;
    }
}

#[derive(Debug)]
/// Auxiliary DSA indexer objective and quality measurements.
pub struct DsaTeacherOutput {
    /// Listwise KL divergence from dense block-attention mass.
    pub loss: Tensor,
    /// Recall of the dense teacher's top-k causal blocks.
    pub top_k_recall: f32,
}

#[derive(Debug, Clone)]
/// Learned block-sparse GQA layer with a compact query/key indexer.
pub struct DsaAttention {
    attention: GroupedQueryAttention,
    index_q: Linear,
    index_k: Linear,
    config: DsaConfig,
    index_dim: usize,
}

struct SparseMask {
    tensor: Tensor,
    selected_blocks: Vec<u32>,
    selected_per_query: usize,
}

impl DsaAttention {
    /// Create a DSA wrapper around an existing grouped-query attention layer.
    pub fn new(
        attention: GroupedQueryAttention,
        index_q: Linear,
        index_k: Linear,
        config: DsaConfig,
        index_dim: usize,
    ) -> Self {
        Self {
            attention,
            index_q,
            index_k,
            config,
            index_dim,
        }
    }

    /// Run DSA inference, optionally updating compact sparse cache state.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        cache: Option<&mut DsaKvCache>,
        seqlen_offset: usize,
        stats: Option<&mut DsaForwardStats>,
    ) -> Result<Tensor> {
        match cache {
            Some(cache) => self.forward_cached(x, rope, cache, seqlen_offset, stats),
            None => self.forward_uncached(x, rope, mask, seqlen_offset, false, stats),
        }
    }

    /// Run the differentiable sparse training path without cache mutation.
    pub fn forward_train(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        stats: Option<&mut DsaForwardStats>,
    ) -> Result<Tensor> {
        self.forward_uncached(x, rope, mask, seqlen_offset, true, stats)
    }

    /// Compute the periodic dense-teacher objective for the learned indexer.
    pub fn teacher_loss(&self, x: &Tensor, rope: &RopeCache) -> Result<DsaTeacherOutput> {
        let (_, seq_len, _) = x.dims3()?;
        let block_count = seq_len.div_ceil(self.config.block_size);
        if block_count <= 1 {
            return Ok(DsaTeacherOutput {
                loss: Tensor::zeros((), DType::F32, x.device())?,
                top_k_recall: 1.0,
            });
        }

        let detached = x.detach();
        let (teacher_q, teacher_k, _) = self.attention.project_train(&detached, rope, 0)?;
        let (index_q, index_k) = self.project_index(&detached, rope, 0, true)?;
        let teacher_probs =
            dense_teacher_block_mass(&teacher_q, &teacher_k, self.config.block_size)?;
        let index_q = pool_blocks(&index_q, self.config.block_size)?.squeeze(2)?;
        let index_k = pool_blocks(&index_k, self.config.block_size)?.squeeze(2)?;

        let causal = block_causal_mask(block_count, x.device())?;
        let index_scores = index_q
            .matmul(&index_k.transpose(1, 2)?.contiguous()?)?
            .affine(1.0 / (self.index_dim as f64).sqrt(), 0.0)?
            .broadcast_add(&causal)?;
        let index_log_probs = candle_nn::ops::log_softmax(&index_scores, 2)?;
        let teacher_log_probs = teacher_probs.affine(1.0, 1e-8)?.log()?;
        let loss = (&teacher_probs * (teacher_log_probs - &index_log_probs)?)?
            .sum_all()?
            .affine(1.0 / (teacher_probs.dim(0)? * block_count) as f64, 0.0)?;
        let top_k_recall = top_k_recall(&teacher_probs, &index_scores, self.config.top_k_blocks)?;
        Ok(DsaTeacherOutput { loss, top_k_recall })
    }

    /// Run sparse attention while recording linear inputs for calibration.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        capture.insert(format!("blocks.{layer_idx}.attn.wq.weight"), x.clone());
        capture.insert(format!("blocks.{layer_idx}.attn.wk.weight"), x.clone());
        capture.insert(format!("blocks.{layer_idx}.attn.wv.weight"), x.clone());
        capture.insert(format!("blocks.{layer_idx}.dsa.index_q.weight"), x.clone());
        capture.insert(format!("blocks.{layer_idx}.dsa.index_k.weight"), x.clone());
        let out = self.forward_uncached(x, rope, mask, 0, false, None)?;
        capture.insert(format!("blocks.{layer_idx}.attn.wo.weight"), out.clone());
        Ok(out)
    }

    /// Return the wrapped grouped-query attention layer.
    pub fn attention(&self) -> &GroupedQueryAttention {
        &self.attention
    }

    /// Return the DSA index-query projection weight.
    pub fn index_q_weight(&self) -> &Tensor {
        self.index_q.weight()
    }

    /// Return the DSA index-key projection weight.
    pub fn index_k_weight(&self) -> &Tensor {
        self.index_k.weight()
    }

    /// Return the sparse attention configuration.
    pub fn config(&self) -> &DsaConfig {
        &self.config
    }

    fn forward_uncached(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        external_mask: Option<&Tensor>,
        seqlen_offset: usize,
        training: bool,
        stats: Option<&mut DsaForwardStats>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let block_count = seq_len.div_ceil(self.config.block_size);
        if self.use_dense(seq_len, block_count) {
            if let Some(stats) = stats {
                stats.queries += batch * seq_len;
                stats.selected_blocks += batch * seq_len * block_count;
                stats.selected_tokens += batch * seq_len * seq_len;
                stats.dense_fallbacks += 1;
            }
            return if training {
                self.attention
                    .forward_train(x, rope, external_mask, seqlen_offset)
            } else {
                self.attention
                    .forward(x, rope, external_mask, None, seqlen_offset)
            };
        }

        let (q, k, v) = if training {
            self.attention.project_train(x, rope, seqlen_offset)?
        } else {
            self.attention.project_inference(x, rope, seqlen_offset)?
        };
        let (index_q, index_k) = self.project_index(x, rope, seqlen_offset, training)?;
        let (mut sparse_mask, local_stats) = build_uncached_mask(
            &index_q,
            &index_k,
            self.config.block_size,
            self.config.top_k_blocks,
        )?;
        if let Some(mask) = external_mask {
            sparse_mask.tensor = sparse_mask.tensor.broadcast_add(mask)?;
        }
        if let Some(stats) = stats {
            stats.merge(&local_stats);
        }
        self.attention.attend_projected_sparse(
            &q,
            &k,
            &v,
            &sparse_mask.tensor,
            &sparse_mask.selected_blocks,
            sparse_mask.selected_per_query,
            self.config.block_size,
            training,
        )
    }

    fn forward_cached(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        cache: &mut DsaKvCache,
        seqlen_offset: usize,
        stats: Option<&mut DsaForwardStats>,
    ) -> Result<Tensor> {
        if cache.seq_len() != seqlen_offset {
            return Err(candle_core::Error::msg(format!(
                "DSA cache length {} does not match sequence offset {seqlen_offset}",
                cache.seq_len()
            )));
        }
        let (batch, seq_len, _) = x.dims3()?;
        let (q, k, v) = self.attention.project_inference(x, rope, seqlen_offset)?;
        let (index_q, index_k) = self.project_index(x, rope, seqlen_offset, false)?;
        let mut outputs = Vec::with_capacity(seq_len);
        let mut aggregate = DsaForwardStats::default();

        for token in 0..seq_len {
            let q_row = q.narrow(1, token, 1)?;
            let k_row = k.narrow(1, token, 1)?;
            let v_row = v.narrow(1, token, 1)?;
            let index_q_row = index_q.narrow(1, token, 1)?;
            let index_k_row = index_k.narrow(1, token, 1)?;
            let (all_k, all_v) = cache.update_kv(&k_row, &v_row)?;
            let total_len = cache.seq_len();
            let block_count = total_len.div_ceil(self.config.block_size);
            let output = if self.use_dense(total_len, block_count) {
                aggregate.queries += batch;
                aggregate.selected_blocks += batch * block_count;
                aggregate.selected_tokens += batch * total_len;
                aggregate.dense_fallbacks += 1;
                let mask = Tensor::zeros((batch, 1, 1, total_len), q.dtype(), q.device())?;
                self.attention
                    .attend_projected(&q_row, &all_k, &all_v, Some(&mask), false)?
            } else {
                let (mask, row_stats) = build_cached_mask(
                    &index_q_row,
                    cache.completed_block_means(),
                    total_len,
                    self.config.block_size,
                    self.config.top_k_blocks,
                )?;
                aggregate.merge(&row_stats);
                self.attention.attend_projected_sparse(
                    &q_row,
                    &all_k,
                    &all_v,
                    &mask.tensor,
                    &mask.selected_blocks,
                    mask.selected_per_query,
                    self.config.block_size,
                    false,
                )?
            };
            outputs.push(output);
            cache.push_index_key(&index_k_row)?;
        }
        if let Some(stats) = stats {
            stats.merge(&aggregate);
        }
        let refs = outputs.iter().collect::<Vec<_>>();
        Tensor::cat(&refs, 1)
    }

    fn project_index(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        seqlen_offset: usize,
        training: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        let q = self
            .index_q
            .forward(x)?
            .reshape((batch, seq_len, 1, self.index_dim))?;
        let k = self
            .index_k
            .forward(x)?
            .reshape((batch, seq_len, 1, self.index_dim))?;
        if training {
            rope.apply(&q, &k, seqlen_offset)
        } else {
            rope.apply_inference(&q, &k, seqlen_offset)
        }
    }

    fn use_dense(&self, seq_len: usize, block_count: usize) -> bool {
        seq_len < self.config.min_seq_len_for_sparsity || block_count <= self.config.top_k_blocks
    }
}

fn dense_teacher_block_mass(q: &Tensor, k: &Tensor, block_size: usize) -> Result<Tensor> {
    let (batch, seq_len, query_heads, head_dim) = q.dims4()?;
    let (_, _, kv_heads, key_dim) = k.dims4()?;
    if key_dim != head_dim || !query_heads.is_multiple_of(kv_heads) {
        return Err(candle_core::Error::msg(
            "DSA teacher requires compatible grouped-query head dimensions",
        ));
    }
    let q_values = q
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let k_values = k
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let blocks = seq_len.div_ceil(block_size);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; batch * blocks * blocks];
    let mut scores = Vec::with_capacity(seq_len);
    for batch_index in 0..batch {
        for query_block in 0..blocks {
            let query_start = query_block * block_size;
            let query_end = (query_start + block_size).min(seq_len);
            let mut rows = 0usize;
            for query_index in query_start..query_end {
                for query_head in 0..query_heads {
                    let kv_head = query_head % kv_heads;
                    let q_offset = ((batch_index * seq_len + query_index) * query_heads
                        + query_head)
                        * head_dim;
                    scores.clear();
                    let mut max_score = f32::NEG_INFINITY;
                    for key_index in 0..=query_index {
                        let k_offset =
                            ((batch_index * seq_len + key_index) * kv_heads + kv_head) * head_dim;
                        let score = q_values[q_offset..q_offset + head_dim]
                            .iter()
                            .zip(&k_values[k_offset..k_offset + head_dim])
                            .map(|(q, k)| q * k)
                            .sum::<f32>()
                            * scale;
                        max_score = max_score.max(score);
                        scores.push(score);
                    }
                    let normalizer = scores
                        .iter()
                        .map(|score| (*score - max_score).exp())
                        .sum::<f32>();
                    for (key_index, score) in scores.iter().enumerate() {
                        let key_block = key_index / block_size;
                        output[(batch_index * blocks + query_block) * blocks + key_block] +=
                            (*score - max_score).exp() / normalizer;
                    }
                    rows += 1;
                }
            }
            let row_scale = 1.0 / rows as f32;
            let offset = (batch_index * blocks + query_block) * blocks;
            output[offset..offset + blocks]
                .iter_mut()
                .for_each(|value| *value *= row_scale);
        }
    }
    Ok(Tensor::from_vec(output, (batch, blocks, blocks), q.device())?.detach())
}

fn pool_blocks(x: &Tensor, block_size: usize) -> Result<Tensor> {
    let (_, seq_len, _, _) = x.dims4()?;
    let mut blocks = Vec::with_capacity(seq_len.div_ceil(block_size));
    for start in (0..seq_len).step_by(block_size) {
        let len = block_size.min(seq_len - start);
        blocks.push(x.narrow(1, start, len)?.mean(1)?);
    }
    let refs = blocks.iter().collect::<Vec<_>>();
    Tensor::stack(&refs, 1)
}

fn block_causal_mask(blocks: usize, device: &candle_core::Device) -> Result<Tensor> {
    let values = (0..blocks)
        .flat_map(|query| (0..blocks).map(move |key| if key <= query { 0.0f32 } else { -1.0e9f32 }))
        .collect::<Vec<_>>();
    Tensor::from_vec(values, (1, blocks, blocks), device)
}

fn build_uncached_mask(
    index_q: &Tensor,
    index_k: &Tensor,
    block_size: usize,
    top_k_blocks: usize,
) -> Result<(SparseMask, DsaForwardStats)> {
    let (batch, seq_len, _, dim) = index_q.dims4()?;
    let q_values = index_q
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let k_values = index_k
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let blocks = seq_len.div_ceil(block_size);
    let mut means = vec![vec![vec![0.0f32; dim]; blocks]; batch];
    for (b, batch_means) in means.iter_mut().enumerate() {
        for (block, mean) in batch_means.iter_mut().enumerate() {
            let start = block * block_size;
            let end = (start + block_size).min(seq_len);
            for token in start..end {
                let base = (b * seq_len + token) * dim;
                for d in 0..dim {
                    mean[d] += k_values[base + d];
                }
            }
            let scale = 1.0 / (end - start) as f32;
            mean.iter_mut().for_each(|value| *value *= scale);
        }
    }

    let mut values = vec![f32::NEG_INFINITY; batch * seq_len * seq_len];
    let mut selected_blocks = vec![u32::MAX; batch * seq_len * top_k_blocks];
    let mut stats = DsaForwardStats::default();
    for b in 0..batch {
        for query in 0..seq_len {
            let current = query / block_size;
            let q_base = (b * seq_len + query) * dim;
            let selected = select_blocks(
                &q_values[q_base..q_base + dim],
                &means[b][..current],
                current,
                top_k_blocks,
            );
            for &block in &selected {
                let start = block * block_size;
                let end = ((block + 1) * block_size).min(query + 1);
                for key in start..end {
                    values[(b * seq_len + query) * seq_len + key] = 0.0;
                    stats.selected_tokens += 1;
                }
            }
            for (slot, block) in selected.iter().enumerate() {
                selected_blocks[(b * seq_len + query) * top_k_blocks + slot] = *block as u32;
            }
            stats.queries += 1;
            stats.selected_blocks += selected.len();
        }
    }
    let mask = Tensor::from_vec(values, (batch, 1, seq_len, seq_len), index_q.device())?
        .to_dtype(index_q.dtype())?;
    Ok((
        SparseMask {
            tensor: mask,
            selected_blocks,
            selected_per_query: top_k_blocks,
        },
        stats,
    ))
}

fn build_cached_mask(
    index_q: &Tensor,
    completed_means: &[Tensor],
    total_len: usize,
    block_size: usize,
    top_k_blocks: usize,
) -> Result<(SparseMask, DsaForwardStats)> {
    let (batch, _, _, dim) = index_q.dims4()?;
    let q_values = index_q
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut means = vec![vec![vec![0.0f32; dim]; completed_means.len()]; batch];
    for (block, tensor) in completed_means.iter().enumerate() {
        let values = tensor
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for b in 0..batch {
            means[b][block].copy_from_slice(&values[b * dim..(b + 1) * dim]);
        }
    }
    let current = (total_len - 1) / block_size;
    let mut mask_values = vec![f32::NEG_INFINITY; batch * total_len];
    let mut selected_blocks = vec![u32::MAX; batch * top_k_blocks];
    let mut stats = DsaForwardStats::default();
    for b in 0..batch {
        let selected = select_blocks(
            &q_values[b * dim..(b + 1) * dim],
            &means[b],
            current,
            top_k_blocks,
        );
        for &block in &selected {
            let start = block * block_size;
            let end = ((block + 1) * block_size).min(total_len);
            for key in start..end {
                mask_values[b * total_len + key] = 0.0;
                stats.selected_tokens += 1;
            }
        }
        for (slot, block) in selected.iter().enumerate() {
            selected_blocks[b * top_k_blocks + slot] = *block as u32;
        }
        stats.queries += 1;
        stats.selected_blocks += selected.len();
    }
    let mask = Tensor::from_vec(mask_values, (batch, 1, 1, total_len), index_q.device())?
        .to_dtype(index_q.dtype())?;
    Ok((
        SparseMask {
            tensor: mask,
            selected_blocks,
            selected_per_query: top_k_blocks,
        },
        stats,
    ))
}

fn select_blocks(
    query: &[f32],
    completed_means: &[Vec<f32>],
    current_block: usize,
    top_k_blocks: usize,
) -> Vec<usize> {
    let mut ranked = completed_means
        .iter()
        .enumerate()
        .filter(|(block, _)| *block < current_block)
        .map(|(block, key)| {
            let score = query.iter().zip(key).map(|(q, k)| q * k).sum::<f32>();
            (block, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_block, left), (right_block, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_block.cmp(right_block))
    });
    ranked.truncate(top_k_blocks.saturating_sub(1));
    let mut selected = ranked
        .into_iter()
        .map(|(block, _)| block)
        .collect::<Vec<_>>();
    selected.push(current_block);
    selected.sort_unstable();
    selected
}

fn top_k_recall(teacher: &Tensor, student: &Tensor, top_k: usize) -> Result<f32> {
    let (batch, blocks, _) = teacher.dims3()?;
    let teacher = teacher
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let student = student
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut hits = 0usize;
    let mut total = 0usize;
    for b in 0..batch {
        for query in 0..blocks {
            let width = query + 1;
            let count = top_k.min(width);
            let start = (b * blocks + query) * blocks;
            let teacher_top = ranked_indices(&teacher[start..start + width], count);
            let student_top = ranked_indices(&student[start..start + width], count);
            hits += teacher_top
                .iter()
                .filter(|index| student_top.contains(index))
                .count();
            total += count;
        }
    }
    Ok(if total == 0 {
        1.0
    } else {
        hits as f32 / total as f32
    })
}

fn ranked_indices(values: &[f32], count: usize) -> Vec<usize> {
    let mut ranked = values.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_unstable_by(|(left_idx, left), (right_idx, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_idx.cmp(right_idx))
    });
    ranked.truncate(count);
    ranked.into_iter().map(|(index, _)| index).collect()
}

#[cfg(test)]
mod tests {
    use super::{select_blocks, top_k_recall};
    use candle_core::{Device, Tensor};

    #[test]
    fn selection_is_chronological_and_current_block_is_mandatory() {
        let selected = select_blocks(
            &[1.0, 0.0],
            &[vec![0.5, 0.0], vec![2.0, 0.0], vec![1.0, 0.0]],
            3,
            2,
        );
        assert_eq!(selected, [1, 3]);
    }

    #[test]
    fn equal_scores_use_lower_block_index() {
        let selected = select_blocks(&[1.0], &[vec![1.0], vec![1.0], vec![1.0]], 3, 3);
        assert_eq!(selected, [0, 1, 3]);
    }

    #[test]
    fn synthetic_indexer_recall_exceeds_eighty_percent() {
        let device = Device::Cpu;
        let values = vec![
            5.0, -1.0e9, -1.0e9, -1.0e9, 1.0, 4.0, -1.0e9, -1.0e9, 1.0, 3.0, 5.0, -1.0e9, 1.0, 2.0,
            4.0, 5.0,
        ];
        let teacher = Tensor::from_vec(values.clone(), (1, 4, 4), &device).unwrap();
        let student = Tensor::from_vec(values, (1, 4, 4), &device).unwrap();
        let recall = top_k_recall(&teacher, &student, 2).unwrap();
        assert!(recall >= 0.8, "recall={recall}");
    }
}
