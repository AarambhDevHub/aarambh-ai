use aarambh_ai_core::{AarambhError, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Quantisation method selector.
pub enum QuantMethod {
    /// Absolute-maximum int8 quantisation.
    Int8Absmax,
    /// Activation-aware int4 quantisation.
    AwqInt4,
    /// GPTQ int4 quantisation.
    GptqInt4,
    /// GGUF Q4_K_M format.
    Q4KM,
    /// GGUF Q5_K_M format.
    Q5KM,
    /// GGUF Q8_0 format.
    Q80,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
/// GGUF output quantisation format.
pub enum GgufFormat {
    /// GGUF Q4_K_M format.
    Q4KM,
    /// GGUF Q5_K_M format.
    Q5KM,
    /// GGUF Q8_0 format.
    Q80,
}

impl GgufFormat {
    /// Parse a GGUF format name.
    pub fn from_name(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "q4_k_m" | "q4km" | "q4" => Ok(Self::Q4KM),
            "q5_k_m" | "q5km" | "q5" => Ok(Self::Q5KM),
            "q8_0" | "q80" | "q8" => Ok(Self::Q80),
            other => Err(AarambhError::Config(format!(
                "unsupported GGUF format '{other}', expected q4_k_m|q5_k_m|q8_0"
            ))),
        }
    }
}

impl QuantMethod {
    /// Parse a quantisation method name.
    pub fn from_name(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "int8" | "absmax" => Ok(Self::Int8Absmax),
            "awq" => Ok(Self::AwqInt4),
            "gptq" => Ok(Self::GptqInt4),
            "q4_k_m" | "q4" => Ok(Self::Q4KM),
            "q5_k_m" | "q5" => Ok(Self::Q5KM),
            "q8_0" | "q8" => Ok(Self::Q80),
            other => Err(AarambhError::Config(format!(
                "unsupported quantisation method '{other}', expected int8|awq|gptq"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Tensor quantised with one signed int8 scale.
pub struct I8QuantizedTensor {
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Quantised int8 values.
    pub data: Vec<i8>,
    /// Scale used to dequantise int8 values.
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Tensor quantised as packed unsigned int4 groups.
pub struct PackedInt4Tensor {
    /// Original tensor shape.
    pub shape: Vec<usize>,
    /// Number of scalar values per quantisation group.
    pub group_size: usize,
    /// Packed 4-bit values.
    pub data: Vec<u8>,
    /// Per-group scales.
    pub scales: Vec<f32>,
    /// Per-group zero points.
    pub zeros: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Supported serialized quantized tensor variants.
pub enum QuantizedTensor {
    /// Signed int8 tensor.
    I8(I8QuantizedTensor),
    /// Packed int4 tensor.
    I4(PackedInt4Tensor),
}

/// Return a tensor's shape as a vector.
pub fn tensor_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.dims().to_vec()
}

/// Return the number of elements described by `shape`.
pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Flatten a tensor to f32 values.
pub fn tensor_to_f32_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

/// Build a tensor from f32 values, shape, and device.
pub fn tensor_from_f32_vec(data: Vec<f32>, shape: &[usize], device: &Device) -> Result<Tensor> {
    let tensor = Tensor::from_vec(data, shape, device)?;
    Ok(tensor)
}

/// Validate that a serialized tensor's value count matches its shape.
pub fn ensure_same_numel(expected_shape: &[usize], len: usize, what: &str) -> Result<()> {
    let expected = numel(expected_shape);
    if expected != len {
        return Err(AarambhError::Shape(format!(
            "{what} has {len} values but shape {expected_shape:?} requires {expected}"
        )));
    }
    Ok(())
}
