use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::Linear;

#[derive(Debug, Clone)]
pub struct SwiGluFfn {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl SwiGluFfn {
    pub fn new(w_gate: Linear, w_up: Linear, w_down: Linear) -> Self {
        Self {
            w_gate,
            w_up,
            w_down,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let hidden = (gate * up)?;
        self.w_down.forward(&hidden)
    }

    pub fn forward_with_capture(
        &self,
        x: &Tensor,
        layer_idx: usize,
        capture: &mut HashMap<String, Tensor>,
    ) -> Result<Tensor> {
        capture.insert(format!("blocks.{layer_idx}.ffn.w_gate.weight"), x.clone());
        capture.insert(format!("blocks.{layer_idx}.ffn.w_up.weight"), x.clone());
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let hidden = (gate * up)?;
        capture.insert(
            format!("blocks.{layer_idx}.ffn.w_down.weight"),
            hidden.clone(),
        );
        self.w_down.forward(&hidden)
    }

    pub fn w_gate_weight(&self) -> &Tensor {
        self.w_gate.weight()
    }

    pub fn w_up_weight(&self) -> &Tensor {
        self.w_up.weight()
    }

    pub fn w_down_weight(&self) -> &Tensor {
        self.w_down.weight()
    }
}
