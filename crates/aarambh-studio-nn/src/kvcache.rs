use candle_core::{DType, Result, Tensor};

use crate::gated_deltanet::DeltaNetState;
use crate::mla::MlaCache;

#[derive(Debug, Clone)]
/// KV state and compact block-index summaries for DSA attention.
pub struct DsaKvCache {
    kv: KVCache,
    completed_block_means: Vec<Tensor>,
    current_block_keys: Vec<Tensor>,
    block_size: usize,
}

impl DsaKvCache {
    /// Create an empty DSA cache with preallocated K/V capacity.
    pub fn with_capacity(capacity: usize, block_size: usize) -> Self {
        Self {
            kv: KVCache::with_capacity(capacity),
            completed_block_means: Vec::with_capacity(capacity.div_ceil(block_size)),
            current_block_keys: Vec::with_capacity(block_size),
            block_size,
        }
    }

    /// Append projected K/V tensors and return the complete causal K/V view.
    pub fn update_kv(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        self.kv.update(k, v)
    }

    /// Record one index-key row after it has been used for causal selection.
    pub fn push_index_key(&mut self, key: &Tensor) -> Result<()> {
        let (batch, seq_len, heads, _) = key.dims4()?;
        if seq_len != 1 || heads != 1 {
            return Err(candle_core::Error::msg(format!(
                "DSA cache expects one index key shaped [batch, 1, 1, dim], got {:?}",
                key.dims()
            )));
        }
        self.current_block_keys
            .push(key.to_dtype(DType::F32)?.reshape((batch, key.dim(3)?))?);
        if self.current_block_keys.len() == self.block_size {
            let rows = self.current_block_keys.iter().collect::<Vec<_>>();
            let mean = Tensor::stack(&rows, 1)?.mean(1)?;
            self.completed_block_means.push(mean);
            self.current_block_keys.clear();
        }
        Ok(())
    }

    /// Return pooled index keys for completed blocks in chronological order.
    pub fn completed_block_means(&self) -> &[Tensor] {
        &self.completed_block_means
    }

    /// Return the number of cached tokens.
    pub fn seq_len(&self) -> usize {
        self.kv.seq_len()
    }

    /// Return the configured K/V capacity.
    pub fn capacity(&self) -> Option<usize> {
        self.kv.capacity()
    }

    /// Return the number of compact completed block summaries.
    pub fn completed_blocks(&self) -> usize {
        self.completed_block_means.len()
    }

    /// Remove all K/V and index state.
    pub fn clear(&mut self) {
        self.kv.clear();
        self.completed_block_means.clear();
        self.current_block_keys.clear();
    }

    /// Truncate within the active partial block.
    ///
    /// Cross-block speculative rollback is performed with exact cache snapshots;
    /// completed blocks intentionally retain only their pooled index key.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        if new_len == 0 {
            self.clear();
            return Ok(());
        }
        let completed_len = self.completed_block_means.len() * self.block_size;
        if new_len < completed_len {
            return Err(candle_core::Error::msg(
                "DSA cache cannot reconstruct a compacted block; restore a cache snapshot",
            ));
        }
        let keep = new_len - completed_len;
        if keep > self.current_block_keys.len() {
            return Err(candle_core::Error::msg(format!(
                "cannot grow DSA cache from {} to {new_len} with truncate",
                self.seq_len()
            )));
        }
        self.current_block_keys.truncate(keep);
        self.kv.truncate(new_len)
    }

    /// Return the maximum number of token rows read by sparse attention.
    pub fn selected_token_limit(&self, top_k_blocks: usize) -> usize {
        top_k_blocks
            .saturating_mul(self.block_size)
            .min(self.seq_len())
    }
}

#[derive(Debug, Clone)]
/// Per-layer cache for full attention, DSA, Gated DeltaNet, or LatentMLA.
pub enum HybridKvCache {
    /// Growing key/value cache used by a full-attention layer.
    Full(KVCache),
    /// Growing K/V cache plus compact DSA index summaries.
    Sparse(DsaKvCache),
    /// Fixed-size recurrent state used by a Gated DeltaNet layer.
    Linear(DeltaNetState),
    /// Compressed-latent cache used by a Multi-Head Latent Attention layer (v4 Phase 41).
    Mla(MlaCache),
}

impl HybridKvCache {
    /// Remove all cached sequence state.
    pub fn clear(&mut self) {
        match self {
            Self::Full(cache) => cache.clear(),
            Self::Sparse(cache) => cache.clear(),
            Self::Linear(state) => state.clear(),
            Self::Mla(cache) => cache.clear(),
        }
    }

    /// Return the number of cached tokens.
    pub fn seq_len(&self) -> usize {
        match self {
            Self::Full(cache) => cache.seq_len(),
            Self::Sparse(cache) => cache.seq_len(),
            Self::Linear(state) => state.seq_len(),
            Self::Mla(cache) => cache.seq_len(),
        }
    }

    /// Truncate a full-attention cache.
    ///
    /// Recurrent caches cannot be reversed and must instead be restored from
    /// a transaction snapshot.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        match self {
            Self::Full(cache) => cache.truncate(new_len),
            Self::Sparse(cache) => cache.truncate(new_len),
            Self::Mla(cache) => cache.truncate(new_len),
            Self::Linear(state) if state.seq_len() == new_len => Ok(()),
            Self::Linear(state) if new_len == 0 => {
                state.clear();
                Ok(())
            }
            Self::Linear(state) => Err(candle_core::Error::msg(format!(
                "cannot truncate Gated DeltaNet state from {} to {new_len}; restore a cache snapshot",
                state.seq_len()
            ))),
        }
    }

    /// Return the full-attention cache when this layer uses GQA.
    pub fn as_full_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            Self::Full(cache) => Some(cache),
            Self::Sparse(_) | Self::Linear(_) | Self::Mla(_) => None,
        }
    }

    /// Return the DSA cache when this layer uses sparse attention.
    pub fn as_sparse_mut(&mut self) -> Option<&mut DsaKvCache> {
        match self {
            Self::Sparse(cache) => Some(cache),
            Self::Full(_) | Self::Linear(_) | Self::Mla(_) => None,
        }
    }

    /// Return the DSA cache when this layer uses sparse attention.
    pub fn as_sparse(&self) -> Option<&DsaKvCache> {
        match self {
            Self::Sparse(cache) => Some(cache),
            Self::Full(_) | Self::Linear(_) | Self::Mla(_) => None,
        }
    }

    /// Return the recurrent state when this layer uses Gated DeltaNet.
    pub fn as_linear_mut(&mut self) -> Option<&mut DeltaNetState> {
        match self {
            Self::Full(_) | Self::Sparse(_) | Self::Mla(_) => None,
            Self::Linear(state) => Some(state),
        }
    }

    /// Return the recurrent state when this layer uses Gated DeltaNet.
    pub fn as_linear(&self) -> Option<&DeltaNetState> {
        match self {
            Self::Full(_) | Self::Sparse(_) | Self::Mla(_) => None,
            Self::Linear(state) => Some(state),
        }
    }

    /// Return the compressed-latent cache when this layer uses LatentMLA.
    pub fn as_mla_mut(&mut self) -> Option<&mut MlaCache> {
        match self {
            Self::Mla(cache) => Some(cache),
            Self::Full(_) | Self::Sparse(_) | Self::Linear(_) => None,
        }
    }

    /// Return the compressed-latent cache when this layer uses LatentMLA.
    pub fn as_mla(&self) -> Option<&MlaCache> {
        match self {
            Self::Mla(cache) => Some(cache),
            Self::Full(_) | Self::Sparse(_) | Self::Linear(_) => None,
        }
    }

    /// Return preallocated full-attention capacity, or `None` for linear state.
    pub fn capacity(&self) -> Option<usize> {
        match self {
            Self::Full(cache) => cache.capacity(),
            Self::Sparse(cache) => cache.capacity(),
            Self::Linear(_) => None,
            Self::Mla(cache) => cache.capacity(),
        }
    }
}

#[derive(Debug, Clone, Default)]
/// Key/value cache for autoregressive attention.
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
    capacity: Option<usize>,
}

impl KVCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            capacity: None,
        }
    }

    /// Create an empty cache that preallocates storage on first update.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            capacity: Some(capacity),
        }
    }

    /// Append key/value tensors and return the full cached tensors.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.capacity.is_some() {
            return self.update_preallocated(k, v);
        }

        let k = match &self.k {
            Some(cached) => Tensor::cat(&[cached, k], 1)?,
            None => k.clone(),
        };
        let v = match &self.v {
            Some(cached) => Tensor::cat(&[cached, v], 1)?,
            None => v.clone(),
        };
        self.len = k.dim(1)?;
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// Remove all cached key/value tensors.
    pub fn clear(&mut self) {
        self.len = 0;
        if self.capacity.is_none() {
            self.k = None;
            self.v = None;
        }
    }

    /// Roll the cache back to a previously committed sequence length.
    pub fn truncate(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.len {
            return Err(candle_core::Error::msg(format!(
                "cannot grow KV cache from {} to {new_len} with truncate",
                self.len
            )));
        }
        if self.capacity.is_some() {
            self.len = new_len;
            return Ok(());
        }
        if new_len == 0 {
            self.clear();
            return Ok(());
        }
        self.k = Some(
            self.k
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("KV cache has no key tensor"))?
                .narrow(1, 0, new_len)?,
        );
        self.v = Some(
            self.v
                .as_ref()
                .ok_or_else(|| candle_core::Error::msg("KV cache has no value tensor"))?
                .narrow(1, 0, new_len)?,
        );
        self.len = new_len;
        Ok(())
    }

    /// Return the cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// Return the preallocated capacity when this cache owns fixed storage.
    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    fn update_preallocated(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let capacity = self.capacity.unwrap_or(0);
        let dims = k.dims();
        if dims.len() != 4 || v.dims() != dims {
            return Err(candle_core::Error::msg(format!(
                "KV cache expects matching rank-4 tensors, got {:?} and {:?}",
                k.dims(),
                v.dims()
            )));
        }
        let seq_len = dims[1];
        if self.len + seq_len > capacity {
            return Err(candle_core::Error::msg(format!(
                "KV cache length {} exceeds capacity {capacity}",
                self.len + seq_len
            )));
        }

        if self.k.is_none() {
            let shape = (dims[0], capacity, dims[2], dims[3]);
            self.k = Some(Tensor::zeros(shape, k.dtype(), k.device())?);
            self.v = Some(Tensor::zeros(shape, v.dtype(), v.device())?);
        }

        let cached_k = self.k.as_ref().unwrap();
        let cached_v = self.v.as_ref().unwrap();
        cached_k.slice_set(&k.contiguous()?, 1, self.len)?;
        cached_v.slice_set(&v.contiguous()?, 1, self.len)?;
        self.len += seq_len;
        Ok((
            cached_k.narrow(1, 0, self.len)?,
            cached_v.narrow(1, 0, self.len)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn values(values: &[f32], device: &Device) -> Tensor {
        Tensor::from_vec(values.to_vec(), (1, values.len(), 1, 1), device).unwrap()
    }

    #[test]
    fn dynamic_cache_truncates_and_regrows() {
        let device = Device::Cpu;
        let mut cache = KVCache::new();
        cache
            .update(&values(&[1.0, 2.0], &device), &values(&[3.0, 4.0], &device))
            .unwrap();
        cache.truncate(1).unwrap();
        let (keys, _) = cache
            .update(&values(&[9.0], &device), &values(&[8.0], &device))
            .unwrap();
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(
            keys.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [1.0, 9.0]
        );
    }

    #[test]
    fn preallocated_cache_overwrites_rolled_back_suffix() {
        let device = Device::Cpu;
        let mut cache = KVCache::with_capacity(4);
        cache
            .update(&values(&[1.0, 2.0], &device), &values(&[3.0, 4.0], &device))
            .unwrap();
        cache.truncate(1).unwrap();
        let (keys, _) = cache
            .update(&values(&[9.0], &device), &values(&[8.0], &device))
            .unwrap();
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(
            keys.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            [1.0, 9.0]
        );
    }

    #[test]
    fn truncate_cannot_grow_cache() {
        let mut cache = KVCache::with_capacity(4);
        assert!(cache.truncate(1).is_err());
    }

    #[test]
    fn dsa_cache_pools_index_keys_in_f32_and_bounds_working_set() {
        let device = Device::Cpu;
        let mut cache = DsaKvCache::with_capacity(32, 2);
        for value in [1.0f32, 3.0] {
            let key = Tensor::full(value, (1, 1, 1, 4), &device)
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let kv = Tensor::zeros((1, 1, 1, 4), DType::F32, &device).unwrap();
            cache.update_kv(&kv, &kv).unwrap();
            cache.push_index_key(&key).unwrap();
        }
        assert_eq!(cache.completed_blocks(), 1);
        assert_eq!(cache.completed_block_means()[0].dtype(), DType::F32);
        assert_eq!(cache.selected_token_limit(1), 2);
        assert_eq!(cache.selected_token_limit(16), cache.seq_len());
    }
}
