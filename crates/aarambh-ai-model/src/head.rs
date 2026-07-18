use aarambh_ai_core::QatTarget;
use aarambh_ai_quant::{QatContext, QatLinear};
use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias};

#[derive(Debug, Clone)]
/// Language-model output projection head.
pub struct LmHead {
    linear: QatLinear,
    tied: bool,
}

impl LmHead {
    /// Create a head tied to the token embedding weight.
    pub fn tied(weight: &Tensor) -> Self {
        Self::tied_with_qat(weight, None)
    }

    /// Create a tied head with optional QAT coverage.
    pub fn tied_with_qat(weight: &Tensor, qat: Option<QatContext>) -> Self {
        Self {
            linear: QatLinear::new(Linear::new(weight.clone(), None), QatTarget::LmHead, qat),
            tied: true,
        }
    }

    /// Create an untied output projection head.
    pub fn untied(hidden_dim: usize, vocab_size: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Self::untied_with_qat(hidden_dim, vocab_size, vb, None)
    }

    /// Create an untied output head with optional QAT coverage.
    pub fn untied_with_qat(
        hidden_dim: usize,
        vocab_size: usize,
        vb: VarBuilder<'_>,
        qat: Option<QatContext>,
    ) -> Result<Self> {
        let linear = linear_no_bias(hidden_dim, vocab_size, vb)?;
        Ok(Self {
            linear: QatLinear::new(linear, QatTarget::LmHead, qat),
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
