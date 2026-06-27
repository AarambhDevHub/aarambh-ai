use candle_core::Module;
use candle_core::{Result, Tensor};
use candle_nn::Linear;

use crate::kvcache::KVCache;
use crate::rope::RopeCache;

#[derive(Debug, Clone)]
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

        let (q, k) = rope.apply(&q, &k, seqlen_offset)?;

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

        let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;

        let attn_weights = match mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let out = attn_weights.matmul(&v)?;

        let out = out.transpose(1, 2)?;
        let out = out.reshape((b, seq_len, self.n_heads * self.head_dim))?;

        self.wo.forward(&out)
    }

    pub fn wq_weight(&self) -> &Tensor {
        self.wq.weight()
    }

    pub fn wk_weight(&self) -> &Tensor {
        self.wk.weight()
    }

    pub fn wv_weight(&self) -> &Tensor {
        self.wv.weight()
    }

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
