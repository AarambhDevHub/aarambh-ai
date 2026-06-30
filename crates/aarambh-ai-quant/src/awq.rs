use aarambh_ai_core::{AarambhError, Result};
use candle_core::Tensor;

use crate::int4::quantise_affine_i4;
use crate::types::{PackedInt4Tensor, tensor_from_f32_vec, tensor_to_f32_vec};

/// Compute per-feature activation scales for AWQ.
pub fn compute_activation_scales(activations: &Tensor) -> Result<Tensor> {
    let dims = activations.dims();
    let features = *dims.last().ok_or_else(|| {
        AarambhError::Shape("activations must have at least one dimension".into())
    })?;
    if features == 0 {
        return Err(AarambhError::Shape(
            "activation feature dimension must be non-zero".into(),
        ));
    }
    let rows = activations.elem_count() / features;
    if rows == 0 {
        return Err(AarambhError::Shape(
            "activations must have at least one row".into(),
        ));
    }
    let values = tensor_to_f32_vec(activations)?;
    let mut scales = vec![0.0f32; features];
    for row in values.chunks(features) {
        for (idx, value) in row.iter().enumerate() {
            scales[idx] += value.abs();
        }
    }
    for scale in &mut scales {
        *scale = (*scale / rows as f32).max(1e-6).sqrt();
    }
    let mean = scales.iter().sum::<f32>() / scales.len() as f32;
    if mean.is_finite() && mean > 0.0 {
        for scale in &mut scales {
            *scale /= mean;
        }
    }
    tensor_from_f32_vec(scales, &[features], activations.device())
}

/// Quantise a rank-2 weight tensor with AWQ int4 scaling.
pub fn quantise_layer_awq(weight: &Tensor, act_scales: &Tensor) -> Result<PackedInt4Tensor> {
    let dims = weight.dims();
    if dims.len() != 2 {
        return Err(AarambhError::Shape(format!(
            "AWQ weight must be rank-2, got {dims:?}"
        )));
    }
    let out = dims[0];
    let input = dims[1];
    if act_scales.dims() != [input] {
        return Err(AarambhError::Shape(format!(
            "AWQ activation scales must have shape [{input}], got {:?}",
            act_scales.dims()
        )));
    }
    let weight_values = tensor_to_f32_vec(weight)?;
    let scale_values = tensor_to_f32_vec(act_scales)?;
    let mut scaled = Vec::with_capacity(weight_values.len());
    for row in 0..out {
        for col in 0..input {
            scaled.push(weight_values[row * input + col] * scale_values[col].max(1e-6));
        }
    }
    let scaled_tensor = tensor_from_f32_vec(scaled, dims, weight.device())?;
    quantise_affine_i4(&scaled_tensor, 128)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn activation_scales_are_positive_and_finite() {
        let device = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, -2., 3., 4., 0., -6.], (2, 3), &device).unwrap();
        let scales = compute_activation_scales(&x).unwrap();
        let values = scales.to_vec1::<f32>().unwrap();
        assert_eq!(values.len(), 3);
        assert!(values.iter().all(|value| value.is_finite() && *value > 0.0));
    }
}
