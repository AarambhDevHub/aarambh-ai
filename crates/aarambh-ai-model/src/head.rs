use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

#[derive(Debug, Clone)]
/// Language-model output projection head.
pub struct LmHead {
    linear: Linear,
    tied: bool,
}

impl LmHead {
    /// Create a head tied to the token embedding weight.
    pub fn tied(weight: &Tensor) -> Self {
        Self {
            linear: Linear::new(weight.clone(), None),
            tied: true,
        }
    }

    /// Create an untied output projection head.
    pub fn untied(hidden_dim: usize, vocab_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let linear = linear_no_bias(hidden_dim, vocab_size, vb)?;
        Ok(Self {
            linear,
            tied: false,
        })
    }

    /// Project hidden states to vocabulary logits.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }

    /// Return the projection weight tensor.
    pub fn weight(&self) -> &Tensor {
        self.linear.weight()
    }

    /// Return true when this head shares embedding weights.
    pub fn is_tied(&self) -> bool {
        self.tied
    }
}
