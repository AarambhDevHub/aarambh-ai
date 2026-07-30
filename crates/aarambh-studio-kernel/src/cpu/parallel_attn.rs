use candle_core::{Error, Result, Shape, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone)]
enum MaskData {
    Matrix {
        data: Vec<f32>,
        kv_seq: usize,
    },
    Broadcast4 {
        data: Vec<f32>,
        batch: usize,
        heads: usize,
        q_seq: usize,
        kv_seq: usize,
    },
}

impl MaskData {
    fn from_tensor(
        mask: &Tensor,
        batch: usize,
        heads: usize,
        q_seq: usize,
        kv_seq: usize,
    ) -> Result<Self> {
        let dims = mask.dims();
        let data = tensor_to_vec_f32(mask)?;
        match dims {
            [mq, mk] if *mq == q_seq && *mk == kv_seq => Ok(Self::Matrix { data, kv_seq }),
            [mb, mh, mq, mk]
                if (*mb == 1 || *mb == batch)
                    && (*mh == 1 || *mh == heads)
                    && *mq == q_seq
                    && *mk == kv_seq =>
            {
                Ok(Self::Broadcast4 {
                    data,
                    batch: *mb,
                    heads: *mh,
                    q_seq,
                    kv_seq,
                })
            }
            _ => Err(Error::msg(format!(
                "unsupported attention mask shape {:?}, expected [{q_seq}, {kv_seq}] or broadcastable [batch, heads, {q_seq}, {kv_seq}]",
                dims
            ))),
        }
    }

    fn value(&self, batch_idx: usize, head_idx: usize, q_idx: usize, kv_idx: usize) -> f32 {
        match self {
            Self::Matrix { data, kv_seq } => data[q_idx * kv_seq + kv_idx],
            Self::Broadcast4 {
                data,
                batch,
                heads,
                q_seq,
                kv_seq,
            } => {
                let b = if *batch == 1 { 0 } else { batch_idx };
                let h = if *heads == 1 { 0 } else { head_idx };
                data[((b * heads + h) * q_seq + q_idx) * kv_seq + kv_idx]
            }
        }
    }
}

/// Compute scaled dot-product attention using Rayon parallelism across rows.
pub fn cpu_parallel_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    attention_impl(q, k, v, mask, false, scale, true)
}

/// Compute causal scaled dot-product attention using Rayon parallelism across rows.
pub fn cpu_parallel_attn_causal(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    attention_impl(q, k, v, None, true, scale, true)
}

/// Compute scaled dot-product attention sequentially for deterministic comparisons.
pub fn cpu_sequential_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    attention_impl(q, k, v, mask, false, scale, false)
}

fn attention_impl(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    causal: bool,
    scale: f64,
    parallel: bool,
) -> Result<Tensor> {
    let (batch, heads, q_seq, head_dim) = q.dims4()?;
    let (k_batch, k_heads, kv_seq, k_head_dim) = k.dims4()?;
    let (v_batch, v_heads, v_seq, v_head_dim) = v.dims4()?;

    if batch != k_batch || batch != v_batch {
        return Err(Error::msg("attention batch dimensions must match"));
    }
    if heads != k_heads || heads != v_heads {
        return Err(Error::msg("attention head dimensions must match"));
    }
    if head_dim != k_head_dim || head_dim != v_head_dim {
        return Err(Error::msg("attention head dimensions must match"));
    }
    if kv_seq != v_seq {
        return Err(Error::msg(
            "attention key/value sequence lengths must match",
        ));
    }
    if head_dim == 0 || kv_seq == 0 {
        return Err(Error::msg("attention head_dim and kv_seq must be non-zero"));
    }

    let q_data = tensor_to_vec_f32(q)?;
    let k_data = tensor_to_vec_f32(k)?;
    let v_data = tensor_to_vec_f32(v)?;
    let mask_data = mask
        .map(|mask| MaskData::from_tensor(mask, batch, heads, q_seq, kv_seq))
        .transpose()?;

    let mut output = vec![0f32; batch * heads * q_seq * head_dim];
    let row_ctx = AttentionRowContext {
        q_data: &q_data,
        k_data: &k_data,
        v_data: &v_data,
        mask: mask_data.as_ref(),
        causal,
        shape: AttentionShape {
            heads,
            q_seq,
            kv_seq,
            head_dim,
        },
        scale: scale as f32,
    };
    if parallel {
        output
            .par_chunks_mut(head_dim)
            .enumerate()
            .for_each(|(row_idx, out_row)| {
                row_ctx.fill(row_idx, out_row);
            });
    } else {
        for (row_idx, out_row) in output.chunks_mut(head_dim).enumerate() {
            row_ctx.fill(row_idx, out_row);
        }
    }

    Tensor::from_vec(
        output,
        Shape::from_dims(&[batch, heads, q_seq, head_dim]),
        q.device(),
    )
}

#[derive(Debug, Clone, Copy)]
struct AttentionShape {
    heads: usize,
    q_seq: usize,
    kv_seq: usize,
    head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
struct AttentionRowContext<'a> {
    q_data: &'a [f32],
    k_data: &'a [f32],
    v_data: &'a [f32],
    mask: Option<&'a MaskData>,
    causal: bool,
    shape: AttentionShape,
    scale: f32,
}

impl AttentionRowContext<'_> {
    fn fill(&self, row_idx: usize, out_row: &mut [f32]) {
        let q_idx = row_idx % self.shape.q_seq;
        let head_idx = (row_idx / self.shape.q_seq) % self.shape.heads;
        let batch_idx = row_idx / (self.shape.heads * self.shape.q_seq);
        let q_offset = ((batch_idx * self.shape.heads + head_idx) * self.shape.q_seq + q_idx)
            * self.shape.head_dim;
        out_row.fill(0.0);
        let mut running_max = f32::NEG_INFINITY;
        let mut running_sum = 0.0f32;
        for kv_idx in 0..self.shape.kv_seq {
            let mask_value = self
                .mask
                .map(|mask| mask.value(batch_idx, head_idx, q_idx, kv_idx))
                .unwrap_or(0.0);
            let causal_masked = self.causal && {
                let shift = self.shape.kv_seq.saturating_sub(self.shape.q_seq);
                kv_idx > shift + q_idx
            };
            if causal_masked || mask_value == f32::NEG_INFINITY {
                continue;
            }
            let k_offset = ((batch_idx * self.shape.heads + head_idx) * self.shape.kv_seq + kv_idx)
                * self.shape.head_dim;
            let mut dot = 0f32;
            for dim_idx in 0..self.shape.head_dim {
                dot += self.q_data[q_offset + dim_idx] * self.k_data[k_offset + dim_idx];
            }
            let score = dot * self.scale + mask_value;
            let next_max = running_max.max(score);
            let old_scale = (running_max - next_max).exp();
            let new_scale = (score - next_max).exp();
            let v_offset = ((batch_idx * self.shape.heads + head_idx) * self.shape.kv_seq + kv_idx)
                * self.shape.head_dim;
            for (dim_idx, out_value) in out_row.iter_mut().enumerate() {
                *out_value = *out_value * old_scale + new_scale * self.v_data[v_offset + dim_idx];
            }
            running_sum = running_sum * old_scale + new_scale;
            running_max = next_max;
        }
        if running_sum > 0.0 {
            let inv_sum = running_sum.recip();
            out_row.iter_mut().for_each(|value| *value *= inv_sum);
        }
    }
}

fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.contiguous()?.flatten_all()?.to_vec1::<f32>()
}
