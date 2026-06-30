use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

#[derive(Debug, Clone)]
/// SwiGLU feed-forward network layer.
pub struct SwiGluFfn {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl SwiGluFfn {
    /// Create a feed-forward layer from gate, up, and down projections.
    pub fn new(w_gate: Linear, w_up: Linear, w_down: Linear) -> Self {
        Self {
            w_gate,
            w_up,
            w_down,
        }
    }

    /// Run the inference feed-forward path, using fused kernels when available.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w_gate.forward(x)?;
        let up = self.w_up.forward(x)?;
        let hidden = aarambh_ai_kernel::fused_ffn::fused_swiglu(&gate, &up).or_else(|_| {
            let gate = candle_nn::ops::silu(&gate)?;
            gate * up
        })?;
        self.w_down.forward(&hidden)
    }

    /// Run the training feed-forward path.
    pub fn forward_train(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let hidden = (gate * up)?;
        self.w_down.forward(&hidden)
    }

    /// Run the feed-forward layer while recording calibration activations.
    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        capture.insert(format!("blocks.{layer_idx}.ffn.w_gate.weight"), x.clone());
        capture.insert(format!("blocks.{layer_idx}.ffn.w_up.weight"), x.clone());
        let gate = self.w_gate.forward(x)?;
        let up = self.w_up.forward(x)?;
        let hidden = aarambh_ai_kernel::fused_ffn::fused_swiglu(&gate, &up).or_else(|_| {
            let gate = candle_nn::ops::silu(&gate)?;
            gate * up
        })?;
        capture.insert(
            format!("blocks.{layer_idx}.ffn.w_down.weight"),
            hidden.clone(),
        );
        self.w_down.forward(&hidden)
    }

    /// Return the gate projection weight tensor.
    pub fn w_gate_weight(&self) -> &Tensor {
        self.w_gate.weight()
    }

    /// Return the up projection weight tensor.
    pub fn w_up_weight(&self) -> &Tensor {
        self.w_up.weight()
    }

    /// Return the down projection weight tensor.
    pub fn w_down_weight(&self) -> &Tensor {
        self.w_down.weight()
    }
}
