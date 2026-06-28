use aarambh_ai_core::{AarambhError, Result};
use candle_core::Tensor;

use crate::types::{tensor_from_f32_vec, tensor_shape, tensor_to_f32_vec};

#[derive(Debug, Clone, Copy)]
pub struct FakeQuantNode {
    pub bits: u8,
    pub symmetric: bool,
}

impl FakeQuantNode {
    pub fn new(bits: u8, symmetric: bool) -> Result<Self> {
        validate_bits(bits)?;
        Ok(Self { bits, symmetric })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.symmetric {
            fake_quantise_symmetric(x, self.bits)
        } else {
            fake_quantise(x, self.bits)
        }
    }
}

pub fn fake_quantise(x: &Tensor, bits: u8) -> Result<Tensor> {
    validate_bits(bits)?;
    let values = tensor_to_f32_vec(x)?;
    let shape = tensor_shape(x);
    let device = x.device();
    let qmax = ((1u32 << bits) - 1) as f32;
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = if (max - min).abs() <= f32::EPSILON {
        1.0
    } else {
        (max - min) / qmax
    };
    let quantised = values
        .into_iter()
        .map(|value| ((value - min) / scale).round().clamp(0.0, qmax) * scale + min)
        .collect();
    tensor_from_f32_vec(quantised, &shape, device)
}

fn fake_quantise_symmetric(x: &Tensor, bits: u8) -> Result<Tensor> {
    validate_bits(bits)?;
    let values = tensor_to_f32_vec(x)?;
    let shape = tensor_shape(x);
    let device = x.device();
    let qmax = ((1u32 << (bits - 1)) - 1) as f32;
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let scale = if max_abs <= f32::EPSILON {
        1.0
    } else {
        max_abs / qmax
    };
    let quantised = values
        .into_iter()
        .map(|value| (value / scale).round().clamp(-qmax, qmax) * scale)
        .collect();
    tensor_from_f32_vec(quantised, &shape, device)
}

fn validate_bits(bits: u8) -> Result<()> {
    if !(2..=8).contains(&bits) {
        return Err(AarambhError::Config(format!(
            "fake quant bits must be in 2..=8, got {bits}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn fake_quant_is_approximately_identity_for_int8() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(
            (0..128)
                .map(|idx| idx as f32 * 0.01 - 0.64)
                .collect::<Vec<_>>(),
            (64, 2),
            &device,
        )
        .unwrap();
        let q = fake_quantise(&x, 8).unwrap();
        let max_err = (x - q)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(max_err < 0.01, "max_err={max_err}");
    }
}
