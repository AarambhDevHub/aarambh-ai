use aarambh_ai_core::Result;
use candle_core::Tensor;

use crate::types::{I8QuantizedTensor, tensor_shape, tensor_to_f32_vec};

pub fn quantise_absmax_i8(tensor: &Tensor) -> Result<I8QuantizedTensor> {
    let values = tensor_to_f32_vec(tensor)?;
    let max_abs = values.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    let scale = if max_abs <= f32::EPSILON {
        1.0
    } else {
        max_abs / 127.0
    };
    let data = values
        .into_iter()
        .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    Ok(I8QuantizedTensor {
        shape: tensor_shape(tensor),
        data,
        scale,
    })
}
