use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Init, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    inner: Embedding,
}

impl TokenEmbedding {
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
