use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::lora::LoraConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub format_version: u32,
    pub model: ModelConfig,
    pub lora: LoraConfig,
    pub base_model: Option<String>,
    pub qlora: bool,
}

impl AdapterMetadata {
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

pub fn load_adapter_weights(varmap: &mut VarMap, dir: impl AsRef<Path>) -> Result<()> {
    let path = dir.as_ref().join("adapter.safetensors");
    varmap.load(path)?;
    Ok(())
}

pub fn adapter_weights_path(dir: impl AsRef<Path>) -> PathBuf {
    dir.as_ref().join("adapter.safetensors")
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}
