use aarambh_studio_core::{AarambhError, Result};
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder, linear};
use serde::{Deserialize, Serialize};

/// Configuration for the trainable vision-to-language projector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ProjectorConfig {
    /// Width of patch embeddings emitted by the vision encoder.
    pub vit_d_model: usize,
    /// Width of the language model hidden states.
    pub llm_d_model: usize,
    /// Multiplier for the hidden layer width.
    pub hidden_mult: usize,
}

impl Default for ProjectorConfig {
    fn default() -> Self {
        Self {
            vit_d_model: 768,
            llm_d_model: 768,
            hidden_mult: 4,
        }
    }
}

impl ProjectorConfig {
    /// Validate projector dimensions.
    pub fn validate(&self) -> Result<()> {
        if self.vit_d_model == 0 || self.llm_d_model == 0 || self.hidden_mult == 0 {
            return Err(AarambhError::Config(
                "projector vit_d_model, llm_d_model, and hidden_mult must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Return the hidden layer width.
    pub fn hidden_dim(&self) -> usize {
        self.llm_d_model * self.hidden_mult
    }
}

/// Two-layer GELU MLP that maps vision patch tokens into LLM token width.
#[derive(Debug, Clone)]
pub struct VisionProjector {
    config: ProjectorConfig,
    fc1: Linear,
    fc2: Linear,
}

impl VisionProjector {
    /// Build a projector from a Candle variable builder.
    pub fn new(config: ProjectorConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        let fc1 = linear(config.vit_d_model, config.hidden_dim(), vb.pp("fc1"))?;
        let fc2 = linear(config.hidden_dim(), config.llm_d_model, vb.pp("fc2"))?;
        Ok(Self { config, fc1, fc2 })
    }

    /// Load a projector checkpoint from SafeTensors.
    pub fn load_safetensors(
        path: impl AsRef<std::path::Path>,
        config: ProjectorConfig,
        device: &candle_core::Device,
        dtype: candle_core::DType,
    ) -> Result<Self> {
        // SAFETY: The checkpoint mapping is read-only and is consumed while the
        // projector creates its owned parameter tensors.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path.as_ref()], dtype, device)? };
        Self::new(config, vb)
    }

    /// Return the projector configuration.
    pub fn config(&self) -> &ProjectorConfig {
        &self.config
    }

    /// Project `[batch, image_tokens, vit_d_model]` into LLM hidden states.
    pub fn forward(&self, patch_tokens: &Tensor) -> Result<Tensor> {
        let dims = patch_tokens.dims();
        if dims.len() != 3 {
            return Err(AarambhError::Shape(format!(
                "patch_tokens must have shape [batch, tokens, vit_d_model], got {dims:?}"
            )));
        }
        if dims[2] != self.config.vit_d_model {
            return Err(AarambhError::Shape(format!(
                "patch token width {} does not match projector vit_d_model {}",
                dims[2], self.config.vit_d_model
            )));
        }
        let hidden = self.fc1.forward(patch_tokens)?.gelu()?;
        Ok(self.fc2.forward(&hidden)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn projector_outputs_llm_width() {
        let device = Device::Cpu;
        let config = ProjectorConfig {
            vit_d_model: 8,
            llm_d_model: 16,
            hidden_mult: 2,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let projector = VisionProjector::new(config, vb).unwrap();
        let input = Tensor::zeros((2, 3, 8), DType::F32, &device).unwrap();
        let output = projector.forward(&input).unwrap();
        assert_eq!(output.dims(), &[2, 3, 16]);
    }
}
