use std::collections::HashMap;

use candle_core::Module;
use candle_core::{Result, Tensor};
use candle_nn::Linear;

use crate::kvcache::KVCache;
use crate::rope::RopeCache;

#[derive(Debug, Clone)]
/// Grouped-query self-attention layer.
pub struct GroupedQueryAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl GroupedQueryAttention {
    /// Create an attention layer from projection layers and head counts.
    pub fn new(
        wq: Linear,
        wk: Linear,
        wv: Linear,
        wo: Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
        }
    }

    pub(crate) fn project_inference(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        let q = self
            .wq
            .forward(x)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let k = self
            .wk
            .forward(x)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let (q, k) = rope.apply_inference(&q, &k, seqlen_offset)?;
        Ok((q, k, v))
    }

    pub(crate) fn project_train(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (batch, seq_len, _) = x.dims3()?;
        let q = self
            .wq
            .forward(x)?
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?;
        let k = self
            .wk
            .forward(x)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?;
        let (q, k) = rope.apply(&q, &k, seqlen_offset)?;
        Ok((q, k, v))
    }

    pub(crate) fn attend_projected(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        training: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _, _) = q.dims4()?;
        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(k, n_repeats)?;
        let v = repeat_heads(v, n_repeats)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let out = if training && mask.is_some() {
            aarambh_ai_kernel::dispatch::attention_forward_candle(&q, &k, &v, mask, self.scale)?
        } else if mask.is_some() {
            // DSA supplies a general additive mask that must not be collapsed
            // into the boolean causal fast-path.
            aarambh_ai_kernel::dispatch::attention_forward_additive(
                &q,
                &k,
                &v,
                mask.expect("checked additive attention mask is present"),
                self.scale,
            )?
        } else if training {
            aarambh_ai_kernel::dispatch::attention_forward_train_causal(&q, &k, &v, self.scale)?
        } else {
            aarambh_ai_kernel::dispatch::attention_forward_causal(&q, &k, &v, self.scale)?
        };
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.n_heads * self.head_dim))?;
        self.wo.forward(&out)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attend_projected_sparse(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        selected_blocks: &[u32],
        selected_per_query: usize,
        block_size: usize,
        training: bool,
    ) -> Result<Tensor> {
        let (batch, seq_len, _, _) = q.dims4()?;
        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(k, n_repeats)?;
        let v = repeat_heads(v, n_repeats)?;
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;
        let out = if training {
            aarambh_ai_kernel::attention_forward_candle(&q, &k, &v, Some(mask), self.scale)?
        } else {
            aarambh_ai_kernel::dsa_sparse_attention_forward(
                &q,
                &k,
                &v,
                mask,
                selected_blocks,
                selected_per_query,
                block_size,
                self.scale,
            )?
        };
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.n_heads * self.head_dim))?;
        self.wo.forward(&out)
    }

    /// Run inference attention, optionally updating a KV cache.
    pub fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let b = dims[0];
        let seq_len = dims[1];

        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;

        let (q, k) = rope.apply_inference(&q, &k, seqlen_offset)?;

        let (k, v) = match kv_cache {
            Some(cache) => cache.update(&k, &v)?,
            None => (k, v),
        };

        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(&k, n_repeats)?;
        let v = repeat_heads(&v, n_repeats)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let out = match mask {
            Some(mask) => {
                aarambh_ai_kernel::dispatch::attention_forward(&q, &k, &v, Some(mask), self.scale)?
            }
            None => aarambh_ai_kernel::dispatch::attention_forward_causal(&q, &k, &v, self.scale)?,
        };

        let out = out.transpose(1, 2)?;
        let out = out.reshape((b, seq_len, self.n_heads * self.head_dim))?;

        self.wo.forward(&out)
    }

    /// Decode one token for multiple independent sequences in a shared projection pass.
    ///
    /// Query, key, value, and output projections are batched. Attention remains
    /// isolated per sequence so ragged KV-cache lengths cannot interact.
    pub fn forward_decode_batch(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        kv_caches: &mut [&mut KVCache],
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 || dims[1] != 1 {
            return Err(candle_core::Error::msg(format!(
                "batched decode expects [batch, 1, hidden], got {dims:?}"
            )));
        }
        let batch = dims[0];
        if kv_caches.len() != batch || seqlen_offsets.len() != batch {
            return Err(candle_core::Error::msg(format!(
                "batched decode received batch {batch}, {} caches, and {} offsets",
                kv_caches.len(),
                seqlen_offsets.len()
            )));
        }

        let q = self
            .wq
            .forward(x)?
            .reshape((batch, 1, self.n_heads, self.head_dim))?;
        let k = self
            .wk
            .forward(x)?
            .reshape((batch, 1, self.n_kv_heads, self.head_dim))?;
        let v = self
            .wv
            .forward(x)?
            .reshape((batch, 1, self.n_kv_heads, self.head_dim))?;

        let n_repeats = self.n_heads / self.n_kv_heads;
        let mut rows = Vec::with_capacity(batch);
        for row in 0..batch {
            let q_row = q.narrow(0, row, 1)?;
            let k_row = k.narrow(0, row, 1)?;
            let v_row = v.narrow(0, row, 1)?;
            let (q_row, k_row) = rope.apply_inference(&q_row, &k_row, seqlen_offsets[row])?;
            let (k_row, v_row) = kv_caches[row].update(&k_row, &v_row)?;
            let k_row = repeat_heads(&k_row, n_repeats)?;
            let v_row = repeat_heads(&v_row, n_repeats)?;
            let q_row = q_row.transpose(1, 2)?.contiguous()?;
            let k_row = k_row.transpose(1, 2)?.contiguous()?;
            let v_row = v_row.transpose(1, 2)?.contiguous()?;
            let out = aarambh_ai_kernel::dispatch::attention_forward_causal(
                &q_row, &k_row, &v_row, self.scale,
            )?;
            rows.push(
                out.transpose(1, 2)?
                    .reshape((1, 1, self.n_heads * self.head_dim))?,
            );
        }

        let row_refs = rows.iter().collect::<Vec<_>>();
        let out = Tensor::cat(&row_refs, 0)?;
        self.wo.forward(&out)
    }

    /// Run the training attention path without mutating a KV cache.
    pub fn forward_train(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let b = dims[0];
        let seq_len = dims[1];

        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;

        let (q, k) = rope.apply(&q, &k, seqlen_offset)?;

        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(&k, n_repeats)?;
        let v = repeat_heads(&v, n_repeats)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let out = match mask {
            Some(mask) => aarambh_ai_kernel::dispatch::attention_forward_train(
                &q,
                &k,
                &v,
                Some(mask),
                self.scale,
            )?,
            None => {
                aarambh_ai_kernel::dispatch::attention_forward_train_causal(&q, &k, &v, self.scale)?
            }
        };

        let out = out.transpose(1, 2)?;
        let out = out.reshape((b, seq_len, self.n_heads * self.head_dim))?;

        self.wo.forward(&out)
    }

    /// Run attention while recording activation tensors for calibration.
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

        let dims = x.dims();
        let b = dims[0];
        let seq_len = dims[1];

        let q = self.wq.forward(x)?;
        let k = self.wk.forward(x)?;
        let v = self.wv.forward(x)?;

        let q = q.reshape((b, seq_len, self.n_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.n_kv_heads, self.head_dim))?;

        let (q, k) = rope.apply_inference(&q, &k, 0)?;

        let n_repeats = self.n_heads / self.n_kv_heads;
        let k = repeat_heads(&k, n_repeats)?;
        let v = repeat_heads(&v, n_repeats)?;

        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        let out = match mask {
            Some(mask) => {
                aarambh_ai_kernel::dispatch::attention_forward(&q, &k, &v, Some(mask), self.scale)?
            }
            None => aarambh_ai_kernel::dispatch::attention_forward_causal(&q, &k, &v, self.scale)?,
        };

        let out = out.transpose(1, 2)?;
        let out = out.reshape((b, seq_len, self.n_heads * self.head_dim))?;
        capture.insert(format!("blocks.{layer_idx}.attn.wo.weight"), out.clone());

        self.wo.forward(&out)
    }

    /// Return the query projection weight tensor.
    pub fn wq_weight(&self) -> &Tensor {
        self.wq.weight()
    }

    /// Return the key projection weight tensor.
    pub fn wk_weight(&self) -> &Tensor {
        self.wk.weight()
    }

    /// Return the value projection weight tensor.
    pub fn wv_weight(&self) -> &Tensor {
        self.wv.weight()
    }

    /// Return the output projection weight tensor.
    pub fn wo_weight(&self) -> &Tensor {
        self.wo.weight()
    }
}

fn repeat_heads(x: &Tensor, n_repeats: usize) -> Result<Tensor> {
    if n_repeats == 1 {
        return Ok(x.clone());
    }
    let dims = x.dims();
    let b = dims[0];
    let seq = dims[1];
    let n_kv = dims[2];
    let head_dim = dims[3];
    let x = x.unsqueeze(2)?;
    let x = x.expand((b, seq, n_repeats, n_kv, head_dim))?;
    x.reshape((b, seq, n_kv * n_repeats, head_dim))?
        .contiguous()
}
