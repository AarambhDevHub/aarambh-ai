use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal,
}

impl Device {
    pub fn to_candle(&self) -> Result<candle_core::Device> {
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            Self::Cuda(i) => Ok(candle_core::Device::new_cuda(*i)?),
            Self::Metal => Ok(candle_core::Device::new_metal(0)?),
        }
    }

    pub fn best_available() -> Self {
        Self::Cpu
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }
}
