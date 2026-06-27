use candle_core::{Result, Tensor};

use crate::attention::GroupedQueryAttention;
use crate::ffn::SwiGluFfn;
use crate::kvcache::KVCache;
use crate::norm::RMSNorm;
use crate::rope::RopeCache;

#[derive(Debug, Clone)]
pub struct TransformerBlock {
    norm1: RMSNorm,
    attn: GroupedQueryAttention,
    norm2: RMSNorm,
    ffn: SwiGluFfn,
}

impl TransformerBlock {
    pub fn new(
        norm1: RMSNorm,
        attn: GroupedQueryAttention,
        norm2: RMSNorm,
        ffn: SwiGluFfn,
    ) -> Self {
        Self {
            norm1,
            attn,
            norm2,
            ffn,
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
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x, rope, mask, kv_cache, seqlen_offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.ffn.forward(&x)?;
        residual + x
    }

    pub fn norm1(&self) -> &RMSNorm {
        &self.norm1
    }

    pub fn attn(&self) -> &GroupedQueryAttention {
        &self.attn
    }

    pub fn norm2(&self) -> &RMSNorm {
        &self.norm2
    }

    pub fn ffn(&self) -> &SwiGluFfn {
        &self.ffn
    }
}
