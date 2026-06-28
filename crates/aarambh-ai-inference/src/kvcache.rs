use aarambh_ai_model::AarambhModel;
use aarambh_ai_nn::KVCache;

#[derive(Debug, Clone)]
pub struct KvCache {
    layers: Vec<KVCache>,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self {
            layers: (0..n_layers).map(|_| KVCache::new()).collect(),
        }
    }

    pub fn for_model(model: &AarambhModel) -> Self {
        Self {
            layers: model.empty_kv_cache(),
        }
    }

    pub fn layers_mut(&mut self) -> &mut [KVCache] {
        &mut self.layers
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn seqlen(&self) -> usize {
        self.layers.first().map(KVCache::seq_len).unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

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
