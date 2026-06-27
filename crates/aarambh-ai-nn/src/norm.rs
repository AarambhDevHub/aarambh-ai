use candle_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Tensor, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        aarambh_ai_kernel::dispatch::rms_norm(x, &self.weight, self.eps)
    }

    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps)
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}
