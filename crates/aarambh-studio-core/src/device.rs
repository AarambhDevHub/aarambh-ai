use crate::error::Result;

#[derive(Debug, Clone, PartialEq)]
/// Logical compute device requested by the user or runtime.
pub enum Device {
    /// Host CPU execution.
    Cpu,
    /// CUDA device by zero-based index.
    Cuda(usize),
    /// Apple Metal device.
    Metal,
}

impl Device {
    /// Convert this logical device to a Candle device handle.
    pub fn to_candle(&self) -> Result<candle_core::Device> {
        match self {
            Self::Cpu => Ok(candle_core::Device::Cpu),
            Self::Cuda(i) => Ok(candle_core::Device::new_cuda(*i)?),
            Self::Metal => Ok(candle_core::Device::new_metal(0)?),
        }
    }

    /// Return the default available device for this build.
    pub fn best_available() -> Self {
        Self::Cpu
    }

    /// Return true when this device is CPU.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }
}
