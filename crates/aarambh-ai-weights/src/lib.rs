use std::path::Path;

pub mod convert;
pub mod gguf;

use aarambh_ai_core::{ModelConfig, Result};
use aarambh_ai_model::AarambhModel;
pub use aarambh_ai_quant::GgufFormat;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub use convert::{HfArch, convert_hf, convert_hf_tensors, convert_hf_with_arch};
pub use gguf::{load_gguf, load_gguf_tensors, load_gguf_with_dtype, save_gguf};

pub fn save_model(model: &AarambhModel, path: impl AsRef<Path>) -> Result<()> {
    candle_core::safetensors::save(&model.named_tensors(), path.as_ref())?;
    Ok(())
}

pub fn load_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    load_model_with_dtype(path, cfg, device, DType::F32)
}

pub fn load_model_with_dtype(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    AarambhModel::new(cfg, vb)
}

pub fn load_any_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    load_any_model_with_dtype(path, cfg, device, DType::F32)
}

pub fn load_any_model_with_dtype(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
        load_gguf_with_dtype(path, device, dtype)
    } else {
        load_model_with_dtype(path, cfg, device, dtype)
    }
}
