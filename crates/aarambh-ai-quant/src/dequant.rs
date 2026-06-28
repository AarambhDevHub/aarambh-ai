use aarambh_ai_core::Result;
use candle_core::{Device, Tensor};

use crate::int4::dequantise_packed_i4_to_vec;
use crate::types::{I8QuantizedTensor, PackedInt4Tensor, ensure_same_numel, tensor_from_f32_vec};

pub fn dequantise_i8(tensor_i8: &I8QuantizedTensor, device: &Device) -> Result<Tensor> {
    ensure_same_numel(&tensor_i8.shape, tensor_i8.data.len(), "i8 tensor")?;
    let values = tensor_i8
        .data
        .iter()
        .map(|value| (*value as f32) * tensor_i8.scale)
        .collect::<Vec<_>>();
    tensor_from_f32_vec(values, &tensor_i8.shape, device)
}

pub fn dequantise_i4(tensor_i4: &PackedInt4Tensor, device: &Device) -> Result<Tensor> {
    let values = dequantise_packed_i4_to_vec(tensor_i4)?;
    tensor_from_f32_vec(values, &tensor_i4.shape, device)
}
