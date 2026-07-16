//! Weight loading, saving, conversion, and GGUF serialization helpers.
#![deny(missing_docs)]

use std::path::Path;

/// HuggingFace conversion helpers.
pub mod convert;
/// GGUF checkpoint reader and writer.
pub mod gguf;

use aarambh_ai_core::{ModelConfig, Result};
use aarambh_ai_model::AarambhModel;
pub use aarambh_ai_quant::GgufFormat;
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

pub use convert::{HfArch, convert_hf, convert_hf_tensors, convert_hf_with_arch};
pub use gguf::{load_gguf, load_gguf_tensors, load_gguf_with_dtype, save_gguf};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Summary of tensors copied and initialized during a hybrid retrofit.
pub struct RetrofitLoadReport {
    /// Number of existing checkpoint tensors copied into the hybrid model.
    pub loaded_tensors: usize,
    /// Number of new Gated DeltaNet tensors left at their fresh initialization.
    pub initialized_deltanet_tensors: usize,
    /// Number of new DSA indexer tensors left at their fresh initialization.
    pub initialized_dsa_tensors: usize,
}

/// Save an Aarambh model as a safetensors checkpoint.
pub fn save_model(model: &AarambhModel, path: impl AsRef<Path>) -> Result<()> {
    candle_core::safetensors::save(&model.named_tensors(), path.as_ref())?;
    Ok(())
}

/// Load a safetensors checkpoint as an Aarambh model using f32 parameters.
pub fn load_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    load_model_with_dtype(path, cfg, device, DType::F32)
}

/// Load a safetensors checkpoint as an Aarambh model using the requested dtype.
pub fn load_model_with_dtype(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
    dtype: DType,
) -> Result<AarambhModel> {
    let path = path.as_ref();
    // SAFETY: Aarambh only reads the checkpoint mapping while constructing owned
    // tensors, and never mutates checkpoint files during this load operation.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device)? };
    AarambhModel::new(cfg, vb)
}

/// Copy a dense SafeTensors checkpoint into an initialized hybrid-model variable map.
///
/// All embedding, normalization, FFN/MoE, output-head, and scheduled full-attention
/// parameters must exist and match shape. Only new `deltanet` and `dsa`
/// parameters may be absent.
pub fn load_retrofit_into_varmap(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    varmap: &mut VarMap,
    device: &Device,
    dtype: DType,
) -> Result<RetrofitLoadReport> {
    if cfg.attention_schedule.is_none() {
        return Err(aarambh_ai_core::AarambhError::Config(
            "retrofit loading requires model.attention_schedule".into(),
        ));
    }
    let source = candle_core::safetensors::load(path.as_ref(), device)?;
    let mut loaded_tensors = 0usize;
    let mut initialized_deltanet_tensors = 0usize;
    let mut initialized_dsa_tensors = 0usize;
    let variables = varmap.data().lock().unwrap();
    for (name, variable) in variables.iter() {
        match source.get(name) {
            Some(value) => {
                if value.dims() != variable.dims() {
                    return Err(aarambh_ai_core::AarambhError::Checkpoint(format!(
                        "retrofit tensor {name} shape {:?} does not match hybrid shape {:?}",
                        value.dims(),
                        variable.dims()
                    )));
                }
                let target_dtype = if name.ends_with(".A_log") || name.ends_with(".dt_bias") {
                    DType::F32
                } else {
                    dtype
                };
                variable.set(&value.to_dtype(target_dtype)?)?;
                loaded_tensors += 1;
            }
            None if name.contains(".deltanet.") => {
                initialized_deltanet_tensors += 1;
            }
            None if name.contains(".dsa.") => {
                initialized_dsa_tensors += 1;
            }
            None => {
                return Err(aarambh_ai_core::AarambhError::Checkpoint(format!(
                    "retrofit source is missing required tensor {name}"
                )));
            }
        }
    }
    drop(variables);
    Ok(RetrofitLoadReport {
        loaded_tensors,
        initialized_deltanet_tensors,
        initialized_dsa_tensors,
    })
}

/// Load either a safetensors or GGUF checkpoint using f32 parameters.
pub fn load_any_model(
    path: impl AsRef<Path>,
    cfg: &ModelConfig,
    device: &Device,
) -> Result<AarambhModel> {
    load_any_model_with_dtype(path, cfg, device, DType::F32)
}

/// Load either a safetensors or GGUF checkpoint using the requested dtype.
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
