use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::lora::LoraConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Metadata saved with a LoRA or QLoRA adapter.
pub struct AdapterMetadata {
    /// Adapter file-format version.
    pub format_version: u32,
    /// Model configuration the adapter was trained against.
    pub model: ModelConfig,
    /// LoRA configuration used for adapter tensors.
    pub lora: LoraConfig,
    /// Optional base model path or identifier.
    pub base_model: Option<String>,
    /// Whether the adapter was trained against a quantized base.
    pub qlora: bool,
}

impl AdapterMetadata {
    /// Create v1 adapter metadata.
    pub fn new(
        model: ModelConfig,
        lora: LoraConfig,
        base_model: Option<String>,
        qlora: bool,
    ) -> Self {
        Self {
            format_version: 1,
            model,
            lora,
            base_model,
            qlora,
        }
    }
}

/// Save adapter metadata and trainable tensors into a directory.
pub fn save_adapter(
    varmap: &VarMap,
    metadata: &AdapterMetadata,
    dir: impl AsRef<Path>,
) -> Result<()> {
    let dir = dir.as_ref();
    fs::create_dir_all(dir)?;
    write_json(dir.join("adapter_config.json"), metadata)?;
    varmap.save(dir.join("adapter.safetensors"))?;
    Ok(())
}

/// Load and validate adapter metadata from a directory.
pub fn load_adapter_metadata(dir: impl AsRef<Path>) -> Result<AdapterMetadata> {
    let dir = dir.as_ref();
    let path = dir.join("adapter_config.json");
    let file = fs::File::open(&path)?;
    let metadata: AdapterMetadata = serde_json::from_reader(file)?;
    if metadata.format_version != 1 {
        return Err(AarambhError::Checkpoint(format!(
            "unsupported adapter format version {} in {}",
            metadata.format_version,
            path.display()
        )));
    }
    metadata.lora.validate()?;
    Ok(metadata)
}

/// Load adapter weights into a variable map.
pub fn load_adapter_weights(varmap: &mut VarMap, dir: impl AsRef<Path>) -> Result<()> {
    let path = dir.as_ref().join("adapter.safetensors");
    varmap.load(path)?;
    Ok(())
}

/// Return the standard adapter weights path for a directory.
pub fn adapter_weights_path(dir: impl AsRef<Path>) -> PathBuf {
    dir.as_ref().join("adapter.safetensors")
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}
