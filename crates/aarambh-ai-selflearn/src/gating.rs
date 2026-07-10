use aarambh_ai_core::{AarambhError, Device, Result};

/// Hardware class used to gate expensive self-learning modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hardware {
    /// Host CPU execution.
    Cpu,
    /// CUDA GPU execution.
    Cuda,
    /// Apple Metal execution.
    Metal,
}

impl From<&Device> for Hardware {
    fn from(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::Cpu,
            Device::Cuda(_) => Self::Cuda,
            Device::Metal => Self::Metal,
        }
    }
}

/// Require CUDA hardware before starting a vision self-learning session.
pub fn require_vision_hardware(device: &Device) -> Result<()> {
    match Hardware::from(device) {
        Hardware::Cuda => Ok(()),
        Hardware::Cpu | Hardware::Metal => Err(AarambhError::Unsupported(
            "Vision self-learning requires Kaggle/CUDA. Text-only self-learning remains supported on CPU; use --self-learn cpu without --image.".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_self_learn_rejects_cpu() {
        let err = require_vision_hardware(&Device::Cpu)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Vision self-learning requires"));
    }

    #[test]
    fn vision_self_learn_allows_cuda() {
        require_vision_hardware(&Device::Cuda(0)).unwrap();
    }
}
