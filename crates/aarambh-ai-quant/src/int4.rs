use aarambh_ai_core::{AarambhError, Result};
use candle_core::Tensor;

use crate::types::{PackedInt4Tensor, ensure_same_numel, numel, tensor_shape, tensor_to_f32_vec};

pub fn quantise_affine_i4(tensor: &Tensor, group_size: usize) -> Result<PackedInt4Tensor> {
    if group_size == 0 {
        return Err(AarambhError::Config("group_size must be non-zero".into()));
    }
    let shape = tensor_shape(tensor);
    let values = tensor_to_f32_vec(tensor)?;
    let mut q_values = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len().div_ceil(group_size));
    let mut zeros = Vec::with_capacity(values.len().div_ceil(group_size));

    for group in values.chunks(group_size) {
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (max - min).abs() <= f32::EPSILON {
            1.0
        } else {
            (max - min) / 15.0
        };
        let zero = (-min / scale).round();
        scales.push(scale);
        zeros.push(zero);
        q_values.extend(
            group
                .iter()
                .map(|value| ((*value / scale) + zero).round().clamp(0.0, 15.0) as u8),
        );
    }

    Ok(PackedInt4Tensor {
        shape,
        group_size,
        data: pack_i4_values(&q_values),
        scales,
        zeros,
    })
}

pub fn pack_i4_values(values: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(values.len().div_ceil(2));
    for pair in values.chunks(2) {
        let lo = pair[0] & 0x0f;
        let hi = pair.get(1).copied().unwrap_or(0) & 0x0f;
        packed.push(lo | (hi << 4));
    }
    packed
}

pub fn unpack_i4_values(data: &[u8], len: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(len);
    for byte in data {
        if values.len() < len {
            values.push(byte & 0x0f);
        }
        if values.len() < len {
            values.push((byte >> 4) & 0x0f);
        }
    }
    values
}

pub fn dequantise_packed_i4_to_vec(tensor: &PackedInt4Tensor) -> Result<Vec<f32>> {
    let n = numel(&tensor.shape);
    let expected_groups = n.div_ceil(tensor.group_size);
    if tensor.group_size == 0 {
        return Err(AarambhError::Config("group_size must be non-zero".into()));
    }
    if tensor.scales.len() != expected_groups || tensor.zeros.len() != expected_groups {
        return Err(AarambhError::Shape(format!(
            "expected {expected_groups} int4 groups, got {} scales and {} zeros",
            tensor.scales.len(),
            tensor.zeros.len()
        )));
    }
    let q_values = unpack_i4_values(&tensor.data, n);
    ensure_same_numel(&tensor.shape, q_values.len(), "int4 tensor")?;
    let mut values = Vec::with_capacity(n);
    for (idx, q) in q_values.into_iter().enumerate() {
        let group = idx / tensor.group_size;
        values.push((q as f32 - tensor.zeros[group]) * tensor.scales[group]);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn pack_unpack_i4_roundtrip_odd_count() {
        let values = vec![0, 1, 15, 7, 9];
        let packed = pack_i4_values(&values);
        assert_eq!(unpack_i4_values(&packed, values.len()), values);
    }

    #[test]
    fn int4_roundtrip_shape_and_error() {
        let device = Device::Cpu;
        let values = (0..31)
            .map(|idx| idx as f32 * 0.03 - 0.5)
            .collect::<Vec<_>>();
        let tensor = Tensor::from_vec(values.clone(), (31,), &device).unwrap();
        let q = quantise_affine_i4(&tensor, 8).unwrap();
        let dq = dequantise_packed_i4_to_vec(&q).unwrap();
        let max_err = values
            .iter()
            .zip(dq.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.04, "max_err={max_err}");
        assert_eq!(q.shape, vec![31]);
    }
}
