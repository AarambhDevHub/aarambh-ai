use candle_core::{Device, Result, Tensor};

#[derive(Debug, Clone)]
pub struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    head_dim: usize,
}

impl RopeCache {
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();

        let mut cos_vals = Vec::with_capacity(max_seq_len * head_dim / 2);
        let mut sin_vals = Vec::with_capacity(max_seq_len * head_dim / 2);

        for pos in 0..max_seq_len {
            for &freq in &inv_freq {
                let val = pos as f32 * freq;
                cos_vals.push(val.cos());
                sin_vals.push(val.sin());
            }
        }

        let shape = (max_seq_len, head_dim / 2);
        let cos = Tensor::from_vec(cos_vals, shape, device)?;
        let sin = Tensor::from_vec(sin_vals, shape, device)?;

        Ok(Self { cos, sin, head_dim })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let q_rot = self.apply_rotate(q, seqlen_offset)?;
        let k_rot = self.apply_rotate(k, seqlen_offset)?;
        Ok((q_rot, k_rot))
    }

    fn apply_rotate(&self, x: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let seq_len = x.dim(1)?;

        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;

        let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

        let x1 = x.narrow(candle_core::D::Minus1, 0, half)?;
        let x2 = x.narrow(candle_core::D::Minus1, half, half)?;

        let rot1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let rot2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;

        Tensor::cat(&[&rot1, &rot2], candle_core::D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn rope_preserves_vector_magnitude() {
        let device = Device::Cpu;
        let rope = RopeCache::new(512, 64, 10000.0, &device).unwrap();
        let q = Tensor::randn(0f32, 1f32, (1, 4, 8, 64), &device).unwrap();
        let (q_rot, _) = rope.apply(&q, &q, 0).unwrap();
        let norm_before: f32 = q.sqr().unwrap().sum_all().unwrap().sqrt().unwrap()
            .to_scalar::<f32>().unwrap();
        let norm_after: f32 = q_rot.sqr().unwrap().sum_all().unwrap().sqrt().unwrap()
            .to_scalar::<f32>().unwrap();
        assert!(
            (norm_before - norm_after).abs() < 1e-4,
            "RoPE changed magnitude: {norm_before} → {norm_after}",
        );
    }
}
