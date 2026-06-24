#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn to_candle(self) -> candle_core::DType {
        match self {
            Self::F32 => candle_core::DType::F32,
            Self::F16 => candle_core::DType::F16,
            Self::BF16 => candle_core::DType::BF16,
        }
    }

    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    Full,
    Half,
    Mixed,
}

impl Precision {
    pub fn weight_dtype(self) -> DType {
        match self {
            Self::Full => DType::F32,
            Self::Half => DType::F16,
            Self::Mixed => DType::BF16,
        }
    }
}
