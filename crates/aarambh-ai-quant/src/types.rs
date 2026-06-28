use aarambh_ai_core::{AarambhError, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuantMethod {
    Int8Absmax,
    AwqInt4,
    GptqInt4,
    Q4KM,
    Q5KM,
    Q80,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GgufFormat {
    Q4KM,
    Q5KM,
    Q80,
}

impl GgufFormat {
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
pub struct I8QuantizedTensor {
    pub shape: Vec<usize>,
    pub data: Vec<i8>,
    pub scale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackedInt4Tensor {
    pub shape: Vec<usize>,
    pub group_size: usize,
    pub data: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizedTensor {
    I8(I8QuantizedTensor),
    I4(PackedInt4Tensor),
}

pub fn tensor_shape(tensor: &Tensor) -> Vec<usize> {
    tensor.dims().to_vec()
}

pub fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn tensor_to_f32_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    Ok(tensor.flatten_all()?.to_vec1::<f32>()?)
}

pub fn tensor_from_f32_vec(data: Vec<f32>, shape: &[usize], device: &Device) -> Result<Tensor> {
    let tensor = Tensor::from_vec(data, shape, device)?;
    Ok(tensor)
}

pub fn ensure_same_numel(expected_shape: &[usize], len: usize, what: &str) -> Result<()> {
    let expected = numel(expected_shape);
    if expected != len {
        return Err(AarambhError::Shape(format!(
            "{what} has {len} values but shape {expected_shape:?} requires {expected}"
        )));
    }
    Ok(())
}
