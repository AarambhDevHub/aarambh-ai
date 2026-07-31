use std::collections::HashMap;

use candle_core::{Result, Tensor};

use crate::attention::GroupedQueryAttention;
use crate::ffn::SwiGluFfn;
use crate::gated_deltanet::GatedDeltaNetLayer;
use crate::kvcache::HybridKvCache;
use crate::mla::MlaAttention;
use crate::moe::{MoeFfn, MoeForwardStats};
use crate::norm::RMSNorm;
use crate::rope::RopeCache;
use crate::sparse_attention::{DsaAttention, DsaForwardStats, DsaTeacherOutput};

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
/// Token-mixing implementation used by one transformer block.
pub enum TokenMixer {
    /// Existing grouped-query full attention.
    Attention(GroupedQueryAttention),
    /// Learned block-sparse grouped-query attention.
    Sparse(DsaAttention),
    /// Fixed-state Gated DeltaNet linear attention.
    GatedDelta(GatedDeltaNetLayer),
    /// Multi-Head Latent Attention with compressed-latent KV cache (v4 Phase 41).
    Mla(MlaAttention),
}

impl TokenMixer {
    fn forward(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        cache: Option<&mut HybridKvCache>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Attention(attn) => {
                let cache = cache
                    .map(|c| {
                        c.as_full_mut().ok_or_else(|| {
                            candle_core::Error::msg(
                                "full-attention block received an incompatible cache",
                            )
                        })
                    })
                    .transpose()?;
                attn.forward(x, rope, mask, cache, seqlen_offset)
            }
            Self::Sparse(attn) => {
                let cache = cache
                    .map(|c| {
                        c.as_sparse_mut().ok_or_else(|| {
                            candle_core::Error::msg("DSA block received an incompatible cache")
                        })
                    })
                    .transpose()?;
                attn.forward(x, rope, mask, cache, seqlen_offset, None)
            }
            Self::GatedDelta(layer) => match cache {
                Some(HybridKvCache::Linear(state)) => layer.forward_cached(x, state),
                None => layer.forward(x),
                Some(_) => Err(candle_core::Error::msg(
                    "Gated DeltaNet block received an incompatible cache",
                )),
            },
            Self::Mla(attn) => {
                let cache = cache
                    .map(|c| {
                        c.as_mla_mut().ok_or_else(|| {
                            candle_core::Error::msg("MLA block received an incompatible cache")
                        })
                    })
                    .transpose()?;
                attn.forward(x, rope, mask, cache, seqlen_offset)
            }
        }
    }

    fn forward_decode_batch(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        caches: &mut [&mut HybridKvCache],
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        match self {
            Self::Attention(attn) => {
                let mut full = caches
                    .iter_mut()
                    .map(|cache| {
                        cache.as_full_mut().ok_or_else(|| {
                            candle_core::Error::msg("full-attention block received a linear cache")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                attn.forward_decode_batch(x, rope, &mut full, seqlen_offsets)
            }
            Self::Sparse(attn) => {
                let mut rows = Vec::with_capacity(caches.len());
                for (row, cache) in caches.iter_mut().enumerate() {
                    let cache = cache.as_sparse_mut().ok_or_else(|| {
                        candle_core::Error::msg("DSA block received an incompatible cache")
                    })?;
                    rows.push(attn.forward(
                        &x.narrow(0, row, 1)?,
                        rope,
                        None,
                        Some(cache),
                        seqlen_offsets[row],
                        None,
                    )?);
                }
                let refs = rows.iter().collect::<Vec<_>>();
                Tensor::cat(&refs, 0)
            }
            Self::GatedDelta(layer) => {
                let mut linear = caches
                    .iter_mut()
                    .map(|cache| {
                        cache.as_linear_mut().ok_or_else(|| {
                            candle_core::Error::msg(
                                "Gated DeltaNet block received an incompatible cache",
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                layer.forward_decode_batch(x, &mut linear)
            }
            Self::Mla(attn) => {
                let mut mla = caches
                    .iter_mut()
                    .map(|cache| {
                        cache.as_mla_mut().ok_or_else(|| {
                            candle_core::Error::msg("MLA block received an incompatible cache")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                attn.forward_decode_batch(x, rope, &mut mla, seqlen_offsets)
            }
        }
    }

    fn forward_train(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Attention(attn) => attn.forward_train(x, rope, mask, seqlen_offset),
            Self::Sparse(attn) => attn.forward_train(x, rope, mask, seqlen_offset, None),
            Self::GatedDelta(layer) => layer.forward_train(x),
            Self::Mla(attn) => attn.forward_train(x, rope, mask, seqlen_offset),
        }
    }

    fn forward_with_capture(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        match self {
            Self::Attention(attn) => attn.forward_with_capture(x, rope, mask, layer_idx, capture),
            Self::Sparse(attn) => attn.forward_with_capture(x, rope, mask, layer_idx, capture),
            Self::GatedDelta(layer) => layer.forward_with_capture(x, layer_idx, capture),
            Self::Mla(attn) => attn.forward_with_capture(x, rope, mask, layer_idx, capture),
        }
    }

    /// Return the full-attention implementation, when selected.
    pub fn as_attention(&self) -> Option<&GroupedQueryAttention> {
        match self {
            Self::Attention(attn) => Some(attn),
            Self::Sparse(attn) => Some(attn.attention()),
            Self::GatedDelta(_) | Self::Mla(_) => None,
        }
    }

    /// Return the DSA implementation, when selected.
    pub fn as_sparse(&self) -> Option<&DsaAttention> {
        match self {
            Self::Sparse(attn) => Some(attn),
            Self::Attention(_) | Self::GatedDelta(_) | Self::Mla(_) => None,
        }
    }

    /// Return the Gated DeltaNet implementation, when selected.
    pub fn as_gated_delta(&self) -> Option<&GatedDeltaNetLayer> {
        match self {
            Self::Attention(_) | Self::Sparse(_) | Self::Mla(_) => None,
            Self::GatedDelta(layer) => Some(layer),
        }
    }

    /// Return the Multi-Head Latent Attention implementation, when selected.
    pub fn as_mla(&self) -> Option<&MlaAttention> {
        match self {
            Self::Mla(attn) => Some(attn),
            Self::Attention(_) | Self::Sparse(_) | Self::GatedDelta(_) => None,
        }
    }

    fn forward_train_with_dsa_stats(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        stats: &mut DsaForwardStats,
    ) -> Result<Tensor> {
        match self {
            Self::Sparse(attn) => attn.forward_train(x, rope, mask, seqlen_offset, Some(stats)),
            _ => self.forward_train(x, rope, mask, seqlen_offset),
        }
    }

    fn dsa_teacher_loss(&self, x: &Tensor, rope: &RopeCache) -> Result<Option<DsaTeacherOutput>> {
        self.as_sparse()
            .map(|attention| attention.teacher_loss(x, rope))
            .transpose()
    }
}

#[derive(Debug, Clone)]
/// Pre-norm transformer decoder block.
pub struct TransformerBlock {
    norm1: RMSNorm,
    mixer: TokenMixer,
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
        Self::new_with_mixer(norm1, TokenMixer::Attention(attn), norm2, ffn)
    }

    /// Create a transformer block from a token mixer and feed-forward implementation.
    pub fn new_with_mixer(
        norm1: RMSNorm,
        mixer: TokenMixer,
        norm2: RMSNorm,
        ffn: FeedForwardLayer,
    ) -> Self {
        Self {
            norm1,
            mixer,
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
        kv_cache: Option<&mut HybridKvCache>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self
            .mixer
            .forward(&x, rope, mask, kv_cache, seqlen_offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.ffn.forward(&x)?;
        residual + x
    }

    /// Decode one token for multiple independent KV-cache sequences.
    pub fn forward_decode_batch(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        kv_caches: &mut [&mut HybridKvCache],
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward(x)?;
        let x = self
            .mixer
            .forward_decode_batch(&x, rope, kv_caches, seqlen_offsets)?;
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
        let x = self.mixer.forward_train(&x, rope, mask, seqlen_offset)?;
        let x = (residual + x)?;

        let residual = x.clone();
        let x = self.norm2.forward_train(&x)?;
        let x = self.ffn.forward_train(&x, None)?;
        residual + x
    }

    /// Run the training block path and collect MoE auxiliary stats.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_train_with_stats(
        &self,
        x: &Tensor,
        rope: &RopeCache,
        mask: Option<&Tensor>,
        seqlen_offset: usize,
        stats: &mut MoeForwardStats,
        dsa_stats: &mut DsaForwardStats,
        collect_dsa_teacher: bool,
        dsa_teachers: &mut Vec<DsaTeacherOutput>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self.norm1.forward_train(x)?;
        if collect_dsa_teacher && let Some(teacher) = self.mixer.dsa_teacher_loss(&x, rope)? {
            dsa_teachers.push(teacher);
        }
        let x =
            self.mixer
                .forward_train_with_dsa_stats(&x, rope, mask, seqlen_offset, dsa_stats)?;
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
            .mixer
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

    /// Return this block's token mixer.
    pub fn mixer(&self) -> &TokenMixer {
        &self.mixer
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
