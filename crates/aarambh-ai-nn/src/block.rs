use std::collections::HashMap;

use candle_core::{Result, Tensor};

use crate::attention::GroupedQueryAttention;
use crate::ffn::SwiGluFfn;
use crate::kvcache::KVCache;
use crate::moe::{MoeFfn, MoeForwardStats};
use crate::norm::RMSNorm;
use crate::rope::RopeCache;

#[derive(Debug, Clone)]
/// Feed-forward implementation used by a transformer block.
pub enum FeedForwardLayer {
    /// Dense SwiGLU feed-forward network.
    Dense(SwiGluFfn),
    /// Mixture-of-Experts SwiGLU feed-forward network.
    Moe(MoeFfn),
}

impl FeedForwardLayer {
    /// Run the inference feed-forward path.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(ffn) => ffn.forward(x),
            Self::Moe(ffn) => ffn.forward(x),
        }
    }

    /// Run the training feed-forward path.
    pub fn forward_train(&self, x: &Tensor, stats: Option<&mut MoeForwardStats>) -> Result<Tensor> {
        match self {
            Self::Dense(ffn) => ffn.forward_train(x),
            Self::Moe(ffn) => ffn.forward_train(x, stats),
        }
    }

    /// Run the feed-forward path while recording calibration activations.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Dense(ffn) => ffn.forward_with_capture(x, layer_idx, capture),
            Self::Moe(ffn) => ffn.forward_with_capture(x, layer_idx, capture),
        }
    }

    /// Return the dense FFN when this layer is dense.
    pub fn as_dense(&self) -> Option<&SwiGluFfn> {
        match self {
            Self::Dense(ffn) => Some(ffn),
            Self::Moe(_) => None,
        }
    }

    /// Return the MoE FFN when this layer is MoE.
    pub fn as_moe(&self) -> Option<&MoeFfn> {
        match self {
            Self::Dense(_) => None,
            Self::Moe(ffn) => Some(ffn),
        }
    }
}

#[derive(Debug, Clone)]
/// Pre-norm transformer decoder block.
pub struct TransformerBlock {
    norm1: RMSNorm,
    attn: GroupedQueryAttention,
    norm2: RMSNorm,
    ffn: FeedForwardLayer,
}

impl TransformerBlock {
    /// Create a transformer block from its norm, attention, and feed-forward layers.
    pub fn new(
        norm1: RMSNorm,
        attn: GroupedQueryAttention,
        norm2: RMSNorm,
        ffn: SwiGluFfn,
    ) -> Self {
        Self::new_with_ffn(norm1, attn, norm2, FeedForwardLayer::Dense(ffn))
    }

    /// Create a transformer block from its norm, attention, and feed-forward implementation.
    pub fn new_with_ffn(
        norm1: RMSNorm,
        attn: GroupedQueryAttention,
        norm2: RMSNorm,
        ffn: FeedForwardLayer,
    ) -> Self {
        Self {
            norm1,
            attn,
            norm2,
            ffn,
        }
    }

    /// Run the inference block path, optionally using a KV cache.
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

    /// Run the training block path without cache mutation.
    pub fn forward_train(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward_train(x)?;
        let x = self.attn.forward_train(&x, rope, mask, seqlen_offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.norm2.forward_train(&x)?;
        let x = self.ffn.forward_train(&x, None)?;
        residual + x
    }

    /// Run the training block path and collect MoE auxiliary stats.
    pub fn forward_train_with_stats(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        stats: &mut MoeForwardStats,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward_train(x)?;
        let x = self.attn.forward_train(&x, rope, mask, seqlen_offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.norm2.forward_train(&x)?;
        let x = self.ffn.forward_train(&x, Some(stats))?;
        residual + x
    }

    /// Run the block while recording activation tensors for calibration.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self
            .attn
            .forward_with_capture(&x, rope, mask, layer_idx, capture)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.ffn.forward_with_capture(&x, layer_idx, capture)?;
        residual + x
    }

    /// Return the first RMSNorm layer.
    pub fn norm1(&self) -> &RMSNorm {
        &self.norm1
    }

    /// Return the attention layer.
    pub fn attn(&self) -> &GroupedQueryAttention {
        &self.attn
    }

    /// Return the second RMSNorm layer.
    pub fn norm2(&self) -> &RMSNorm {
        &self.norm2
    }

    /// Return the feed-forward layer.
    pub fn ffn(&self) -> &FeedForwardLayer {
        &self.ffn
    }
}
