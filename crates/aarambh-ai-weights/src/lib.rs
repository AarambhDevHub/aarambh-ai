use std::path::Path;

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use aarambh_ai_model::AarambhModel;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub fn save_model(model: &AarambhModel, path: impl AsRef<Path>) -> Result<()> {
    candle_core::safetensors::save(&model.named_tensors(), path.as_ref())?;
    Ok(())
}

pub fn load_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
    AarambhModel::new(cfg, vb)
}

pub fn convert_hf(_hf_dir: &Path, _cfg: &ModelConfig) -> Result<AarambhModel> {
    Err(AarambhError::Unsupported(
        "HuggingFace conversion is planned for Phase 8".into(),
    ))
}
