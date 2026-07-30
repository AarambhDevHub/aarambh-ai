use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder};

#[derive(Debug, Clone)]
/// Token embedding layer for Aarambh decoder models.
pub struct TokenEmbedding {
    inner: Embedding,
}

impl TokenEmbedding {
    /// Create an embedding table with normal initialization.
    pub fn new(vocab_size: usize, hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let weight = vb.get_with_hints(
            (vocab_size, hidden_dim),
            "weight",
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;
        let inner = Embedding::new(weight, hidden_dim);
        Ok(Self { inner })
    }

    /// Convert token ids to hidden states.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.inner.forward(ids)
    }

    /// Return the embedding weight tensor.
    pub fn weight(&self) -> &Tensor {
        self.inner.embeddings()
    }

    /// Return the embedding hidden width.
    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
}
