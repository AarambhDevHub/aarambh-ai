use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Device, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_ai_data::DataLoader;
use aarambh_ai_data::dataset::PlaintextDataset;
use aarambh_ai_tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};

use crate::trainer::Trainer;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingRunConfig {
    pub dataset_path: PathBuf,
    pub tokenizer_path: Option<PathBuf>,
    pub tokenizer_save_path: Option<PathBuf>,
    pub vocab_size: usize,
    pub validation_split: f64,
    pub shuffle: bool,
    pub resume: bool,
    pub device: String,
    pub model: ModelConfig,
    pub train: TrainConfig,
}

impl Default for TrainingRunConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::new(),
            tokenizer_path: None,
            tokenizer_save_path: None,
            vocab_size: 32000,
            validation_split: 0.05,
            shuffle: true,
            resume: false,
            device: "cpu".to_string(),
            model: ModelConfig::tiny(),
            train: TrainConfig::default(),
        }
    }
}

impl TrainingRunConfig {
    pub fn from_toml(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        toml::from_str(&content).map_err(|err| {
            AarambhError::Config(format!(
                "failed to parse training config {}: {err}",
                path.as_ref().display()
            ))
        })
    }

    pub fn device(&self) -> Result<Device> {
        let value = self.device.trim().to_ascii_lowercase();
        match value.as_str() {
            "cpu" => Ok(Device::Cpu),
            "metal" => Ok(Device::Metal),
            value if value.starts_with("cuda") => {
                let index = value
                    .split_once(':')
                    .map(|(_, index)| index.parse::<usize>())
                    .transpose()
                    .map_err(|err| AarambhError::Config(format!("invalid CUDA device: {err}")))?
                    .unwrap_or(0);
                Ok(Device::Cuda(index))
            }
            other => Err(AarambhError::Config(format!(
                "unsupported training device '{other}'"
            ))),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.dataset_path.as_os_str().is_empty() {
            return Err(AarambhError::Config("dataset_path is required".into()));
        }
        if !(0.0..1.0).contains(&self.validation_split) {
            return Err(AarambhError::Config(
                "validation_split must be in [0, 1)".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(AarambhError::Config("vocab_size must be non-zero".into()));
        }
        Ok(())
    }
}

pub fn run_training_from_config(path: impl AsRef<Path>) -> Result<()> {
    let config = TrainingRunConfig::from_toml(path)?;
    config.validate()?;

    let device = config.device()?;
    let candle_device = device.to_candle()?;
    let tokenizer = prepare_tokenizer(&config)?;
    let mut model_config = config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();

    let (train_dataset, val_dataset) = load_datasets(&config)?;
    let train_loader = DataLoader::new(
        &train_dataset,
        &tokenizer,
        config.train.batch_size,
        model_config.max_seq_len,
        config.shuffle,
        device.clone(),
    );
    let val_loader = val_dataset.map(|dataset| {
        DataLoader::new(
            &dataset,
            &tokenizer,
            config.train.batch_size,
            model_config.max_seq_len,
            false,
            device.clone(),
        )
    });

    let mut trainer = Trainer::new(
        model_config,
        config.train.clone(),
        train_loader,
        val_loader,
        candle_device,
    )?;
    if config.resume && trainer.load_latest_checkpoint()? {
        println!("resumed checkpoint at step={}", trainer.state().step);
    }
    trainer.train()
}

fn prepare_tokenizer(config: &TrainingRunConfig) -> Result<BpeTokenizer> {
    if let Some(path) = &config.tokenizer_path {
        return BpeTokenizer::from_pretrained(path);
    }

    fs::create_dir_all(&config.train.checkpoint_dir)?;
    let save_path = config
        .tokenizer_save_path
        .clone()
        .unwrap_or_else(|| config.train.checkpoint_dir.join("tokenizer.json"));
    if save_path.exists() {
        return BpeTokenizer::from_pretrained(save_path);
    }

    let tokenizer = BpeTokenizer::train(&config.dataset_path, config.vocab_size)?;
    tokenizer.save_pretrained(save_path)?;
    Ok(tokenizer)
}

fn load_datasets(
    config: &TrainingRunConfig,
) -> Result<(PlaintextDataset, Option<PlaintextDataset>)> {
    let content = fs::read_to_string(&config.dataset_path)?;
    let mut lines = content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if lines.is_empty() && !content.is_empty() {
        lines.push(content);
    }
    if lines.is_empty() {
        return Err(AarambhError::Config(format!(
            "dataset {} is empty",
            config.dataset_path.display()
        )));
    }

    let val_count = ((lines.len() as f64) * config.validation_split).round() as usize;
    let val_count = val_count.min(lines.len().saturating_sub(1));
    let split_at = lines.len() - val_count;
    let val_lines = if val_count > 0 {
        Some(lines.split_off(split_at))
    } else {
        None
    };

    let train = PlaintextDataset::from_lines(lines);
    let val = val_lines.map(PlaintextDataset::from_lines);
    Ok((train, val))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_architecture_adamw_beta2() {
        let config = TrainingRunConfig::default();
        assert_eq!(config.train.beta2, 0.95);
    }

    #[test]
    fn parses_cpu_device() {
        let config = TrainingRunConfig::default();
        assert_eq!(config.device().unwrap(), Device::Cpu);
    }
}
