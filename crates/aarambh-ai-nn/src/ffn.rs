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
}
