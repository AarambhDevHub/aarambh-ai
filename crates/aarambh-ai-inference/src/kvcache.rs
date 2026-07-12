use aarambh_ai_model::AarambhModel;
use aarambh_ai_nn::KVCache;
use candle_core::Result;

#[derive(Debug, Clone)]
/// Multi-layer KV cache used by the inference engine.
pub struct KvCache {
    layers: Vec<KVCache>,
}

impl KvCache {
    /// Create a cache with `n_layers` empty layer caches.
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| KVCache::new()).collect(),
        }
    }

    /// Create a cache sized for a model.
    pub fn for_model(model: &AarambhModel) -> Self {
        Self {
            layers: model.empty_kv_cache(),
        }
    }

    /// Create a model-sized cache with fixed sequence capacity per layer.
    pub fn for_model_with_capacity(model: &AarambhModel, capacity: usize) -> Self {
        let n_layers = model.empty_kv_cache().len();
        Self {
            layers: (0..n_layers)
                .map(|_| KVCache::with_capacity(capacity))
                .collect(),
        }
    }

    /// Return mutable layer caches.
    pub fn layers_mut(&mut self) -> &mut [KVCache] {
        &mut self.layers
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    /// Roll every layer back to a previously committed sequence length.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        for layer in &mut self.layers {
            layer.truncate(new_len)?;
        }
        debug_assert!(self.layers.iter().all(|layer| layer.seq_len() == new_len));
        Ok(())
    }

    /// Return cached sequence length from the first layer.
    pub fn seqlen(&self) -> usize {
        self.layers.first().map(KVCache::seq_len).unwrap_or(0)
    }

    /// Return the number of layer caches.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Return true when no layers are cached.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn kvcache_seqlen_grows_each_step() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();
        let k1 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        let v1 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        cache.update(&k1, &v1).unwrap();
        assert_eq!(cache.seq_len(), 1);

        let k2 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        let v2 = Tensor::zeros((1, 1, 2, 64), candle_core::DType::F32, &device).unwrap();
        cache.update(&k2, &v2).unwrap();
        assert_eq!(cache.seq_len(), 2);
    }
}
