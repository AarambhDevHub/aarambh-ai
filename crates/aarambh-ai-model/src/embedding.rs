use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};

#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    inner: Embedding,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, hidden_dim: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let inner = embedding(vocab_size, hidden_dim, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.inner.forward(ids)
    }

    pub fn weight(&self) -> &Tensor {
        self.inner.embeddings()
    }

    pub fn hidden_size(&self) -> usize {
        self.inner.hidden_size()
    }
}
