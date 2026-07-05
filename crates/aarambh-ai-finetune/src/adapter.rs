use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::lora::LoraConfig;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Adapter algorithm stored in an adapter directory.
pub enum AdapterMethod {
    /// Classic low-rank adaptation.
    #[default]
    Lora,
    /// Weight-decomposed low-rank adaptation.
    Dora,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Metadata saved with a LoRA, QLoRA, DoRA, or QDoRA adapter.
pub struct AdapterMetadata {
    /// Adapter file-format version.
    pub format_version: u32,
    /// Adapter algorithm.
    #[serde(default)]
    pub method: AdapterMethod,
    /// Model configuration the adapter was trained against.
    pub model: ModelConfig,
    /// Low-rank adapter configuration used for adapter tensors.
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
        Self::new_with_method(model, lora, base_model, qlora, AdapterMethod::Lora)
    }

    /// Create v1 adapter metadata with an explicit adapter method.
    pub fn new_with_method(
        model: ModelConfig,
        lora: LoraConfig,
        base_model: Option<String>,
        qlora: bool,
        method: AdapterMethod,
    ) -> Self {
        Self {
            format_version: 1,
            method,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_adapter_metadata_defaults_to_lora_method() {
        let json = serde_json::json!({
            "format_version": 1,
            "model": ModelConfig::tiny(),
            "lora": LoraConfig::default(),
            "base_model": null,
            "qlora": false
        });
        let metadata: AdapterMetadata = serde_json::from_value(json).unwrap();
        assert_eq!(metadata.method, AdapterMethod::Lora);
    }

    #[test]
    fn dora_metadata_records_adapter_method() {
        let metadata = AdapterMetadata::new_with_method(
            ModelConfig::tiny(),
            LoraConfig::default(),
            Some("base.safetensors".into()),
            true,
            AdapterMethod::Dora,
        );
        assert_eq!(metadata.method, AdapterMethod::Dora);
        assert!(metadata.qlora);
    }
}
