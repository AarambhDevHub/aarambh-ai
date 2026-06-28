use aarambh_ai_core::{AarambhError, Result};
use candle_core::{Device, Tensor};

use crate::absmax::quantise_absmax_i8;
use crate::dequant::dequantise_i8;
use crate::types::I8QuantizedTensor;

#[derive(Debug, Clone, Default)]
struct QuantisedKvLayer {
    k: Option<I8QuantizedTensor>,
    v: Option<I8QuantizedTensor>,
}

#[derive(Debug, Clone)]
pub struct QuantisedKvCache {
    layers: Vec<QuantisedKvLayer>,
    device: Device,
}

impl QuantisedKvCache {
    pub fn new(n_layers: usize, _n_kv_heads: usize, _head_dim: usize, device: &Device) -> Self {
        Self {
            layers: vec![QuantisedKvLayer::default(); n_layers],
            device: device.clone(),
        }
    }

    pub fn append_and_get(
        &mut self,
        layer: usize,
        k: &Tensor,
        v: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let n_layers = self.layers.len();
        let cache = self.layers.get_mut(layer).ok_or_else(|| {
            AarambhError::Shape(format!(
                "quantized KV cache layer {layer} out of range for {} layers",
                n_layers
            ))
        })?;

        let k_full = match &cache.k {
            Some(existing) => Tensor::cat(&[dequantise_i8(existing, &self.device)?, k.clone()], 1)?,
            None => k.clone(),
        };
        let v_full = match &cache.v {
            Some(existing) => Tensor::cat(&[dequantise_i8(existing, &self.device)?, v.clone()], 1)?,
            None => v.clone(),
        };
        cache.k = Some(quantise_absmax_i8(&k_full)?);
        cache.v = Some(quantise_absmax_i8(&v_full)?);
        Ok((k_full, v_full))
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.k = None;
            layer.v = None;
        }
    }

    pub fn seq_len(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .and_then(|cache| cache.k.as_ref())
            .and_then(|k| k.shape.get(1).copied())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn quantized_kv_cache_sequence_grows() {
        let device = Device::Cpu;
        let mut cache = QuantisedKvCache::new(1, 1, 4, &device);
        let k1 = Tensor::from_vec(vec![0.1f32; 4], (1, 1, 1, 4), &device).unwrap();
        let v1 = Tensor::from_vec(vec![0.2f32; 4], (1, 1, 1, 4), &device).unwrap();
        cache.append_and_get(0, &k1, &v1).unwrap();
        assert_eq!(cache.seq_len(0), 1);
        cache.append_and_get(0, &k1, &v1).unwrap();
        assert_eq!(cache.seq_len(0), 2);
    }
}
