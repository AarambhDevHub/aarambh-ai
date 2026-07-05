use aarambh_ai_core::ModelConfig;
use candle_core::{DType, Device, Error, Result, Tensor};

use crate::rope_scaling::{base_inverse_frequencies, inverse_frequencies_for_config};

#[derive(Debug, Clone)]
/// Precomputed rotary-position embedding cache.
pub struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
    max_seq_len: usize,
}

impl RopeCache {
    /// Build cosine and sine tables from a model config, including optional RoPE scaling.
    pub fn from_config(cfg: &ModelConfig, dtype: DType, device: &Device) -> Result<Self> {
        let (inv_freq, attn_factor) =
            inverse_frequencies_for_config(cfg).map_err(|err| Error::msg(err.to_string()))?;
        Self::from_inverse_frequencies(
            cfg.max_seq_len,
            cfg.head_dim(),
            &inv_freq,
            attn_factor,
            dtype,
            device,
        )
    }

    /// Build cosine and sine tables for the requested context length and head width.
    pub fn new(
        max_seq_len: usize,
        head_dim: usize,
        theta: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq =
            base_inverse_frequencies(head_dim, theta).map_err(|err| Error::msg(err.to_string()))?;
        Self::from_inverse_frequencies(max_seq_len, head_dim, &inv_freq, 1.0, dtype, device)
    }

    fn from_inverse_frequencies(
        max_seq_len: usize,
        head_dim: usize,
        inv_freq: &[f32],
        attn_factor: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let mut cos_vals = Vec::with_capacity(max_seq_len * head_dim / 2);
        let mut sin_vals = Vec::with_capacity(max_seq_len * head_dim / 2);

        for pos in 0..max_seq_len {
            for &freq in inv_freq {
                let val = pos as f32 * freq;
                cos_vals.push(val.cos() * attn_factor);
                sin_vals.push(val.sin() * attn_factor);
            }
        }

        let shape = (max_seq_len, head_dim / 2);
        let cos = Tensor::from_vec(cos_vals, shape, device)?.to_dtype(dtype)?;
        let sin = Tensor::from_vec(sin_vals, shape, device)?.to_dtype(dtype)?;

        Ok(Self {
            cos,
            sin,
            head_dim,
            max_seq_len,
        })
    }

    /// Return the cached context length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Return the RoPE head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Apply RoPE to query and key tensors through the standard path.
    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let q_rot = self.apply_rotate(q, seqlen_offset)?;
        let k_rot = self.apply_rotate(k, seqlen_offset)?;
        Ok((q_rot, k_rot))
    }

    /// Apply RoPE to query and key tensors, using fused kernels when available.
    pub fn apply_inference(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let q_rot = self
            .apply_rotate_fused(q, seqlen_offset)
            .or_else(|_| self.apply_rotate(q, seqlen_offset))?;
        let k_rot = self
            .apply_rotate_fused(k, seqlen_offset)
            .or_else(|_| self.apply_rotate(k, seqlen_offset))?;
        Ok((q_rot, k_rot))
    }

    fn apply_rotate(&self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let seq_len = x.dim(1)?;

        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = cos.to_dtype(x.dtype())?;
        let sin = sin.to_dtype(x.dtype())?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = x.narrow(candle_core::D::Minus1, half, half)?;

        let rot1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rot2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&rot1, &rot2], candle_core::D::Minus1)
    }

    fn apply_rotate_fused(&self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let cos = self.cos.to_dtype(x.dtype())?;
        let sin = self.sin.to_dtype(x.dtype())?;
        if seqlen_offset + seq_len > cos.dim(0)? {
            return self.apply_rotate(x, seqlen_offset);
        }
        aarambh_ai_kernel::fused_rope::fused_rope_apply(x, &cos, &sin, seqlen_offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rope_preserves_vector_magnitude() {
        let device = Device::Cpu;
        let rope = RopeCache::new(512, 64, 10000.0, DType::F32, &device).unwrap();
        let q = Tensor::randn(0f32, 1f32, (1, 4, 8, 64), &device).unwrap();
        let (q_rot, _) = rope.apply(&q, &q, 0).unwrap();
        let norm_before: f32 = q
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let norm_after: f32 = q_rot
            .sqr()
            .unwrap()
            .sum_all()
            .unwrap()
            .sqrt()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE changed magnitude: {norm_before} → {norm_after}",
        );
    }
}
