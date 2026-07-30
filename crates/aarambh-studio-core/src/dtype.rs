use std::fmt;
use std::str::FromStr;

use crate::error::AarambhError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Numeric dtype used for model weights and activation tensors.
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// IEEE 16-bit floating point.
    F16,
    /// Brain floating point 16-bit.
    BF16,
}

impl DType {
    /// Convert this dtype to Candle's dtype representation.
    pub fn to_candle(self) -> candle_core::DType {
        match self {
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::BF16 => candle_core::DType::BF16,
        }
    }

    /// Return the number of bytes used by one scalar of this dtype.
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

impl FromStr for DType {
    type Err = AarambhError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" | "full" => Ok(Self::F32),
            "f16" | "float16" | "half" => Ok(Self::F16),
            "bf16" | "bfloat16" | "mixed" => Ok(Self::BF16),
            other => Err(AarambhError::Config(format!(
                "unsupported dtype '{other}', expected f32|f16|bf16|mixed"
            ))),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::BF16 => "bf16",
        };
        f.write_str(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// High-level precision preset.
pub enum Precision {
    /// Full f32 weights.
    Full,
    /// Half precision f16 weights.
    Half,
    /// Mixed precision using bf16 weights.
    Mixed,
}

impl Precision {
    /// Return the default weight dtype for this precision preset.
    pub fn weight_dtype(self) -> DType {
        match self {
            Self::Full => DType::F32,
            Self::Half => DType::F16,
            Self::Mixed => DType::BF16,
        }
    }
}
