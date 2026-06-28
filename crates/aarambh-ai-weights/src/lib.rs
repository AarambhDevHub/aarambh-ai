use std::path::Path;

pub mod convert;
pub mod gguf;

use aarambh_ai_core::{ModelConfig, Result};
use aarambh_ai_model::AarambhModel;
pub use aarambh_ai_quant::GgufFormat;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub use convert::{HfArch, convert_hf, convert_hf_tensors, convert_hf_with_arch};
pub use gguf::{load_gguf, load_gguf_tensors, save_gguf};

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

pub fn load_any_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    if path.extension().and_then(|ext| ext.to_str()) == Some("gguf") {
        load_gguf(path, device)
    } else {
        load_model(path, cfg, device)
    }
}
