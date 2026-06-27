use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

#[derive(Debug, Clone)]
pub struct LmHead {
    linear: Linear,
    tied: bool,
}

impl LmHead {
    pub fn tied(weight: &Tensor) -> Self {
        Self {
            linear: Linear::new(weight.clone(), None),
            tied: true,
        }
    }

    pub fn untied(hidden_dim: usize, vocab_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let linear = linear_no_bias(hidden_dim, vocab_size, vb)?;
        Ok(Self {
            linear,
            tied: false,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear.forward(x)
    }

    pub fn weight(&self) -> &Tensor {
        self.linear.weight()
    }

    pub fn is_tied(&self) -> bool {
        self.tied
    }
}
