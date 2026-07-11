use candle_core::{Result, Tensor};

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
}
