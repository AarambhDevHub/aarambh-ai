use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
/// Root-mean-square normalization layer.
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    /// Create an RMSNorm layer from weight tensor and epsilon.
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Run inference RMSNorm through kernel dispatch.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        aarambh_ai_kernel::dispatch::rms_norm(x, &self.weight, self.eps)
    }

    /// Run training RMSNorm through Candle.
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps)
    }

    /// Return the normalization weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}
