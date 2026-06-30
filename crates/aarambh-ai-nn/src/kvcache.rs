use candle_core::{Result, Tensor};

#[derive(Debug, Clone, Default)]
/// Key/value cache for autoregressive attention.
pub struct KVCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KVCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Append key/value tensors and return the full cached tensors.
    pub fn update(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = match &self.k {
            Some(cached) => Tensor::cat(&[cached, k], 1)?,
            None => k.clone(),
        };
        let v = match &self.v {
            Some(cached) => Tensor::cat(&[cached, v], 1)?,
            None => v.clone(),
        };
        self.k = Some(k.clone());
        self.v = Some(v.clone());
        Ok((k, v))
    }

    /// Remove all cached key/value tensors.
    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// Return the cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(1).unwrap_or(0)).unwrap_or(0)
    }
}
