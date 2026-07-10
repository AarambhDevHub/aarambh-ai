use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use aarambh_ai_core::{
    AarambhError, DType as AarambhDType, Device, ModelConfig, Result, TokenizerLike, TrainConfig,
};
use aarambh_ai_data::dataset::PlaintextDataset;
use aarambh_ai_data::{DataLoader, DataShard};
use aarambh_ai_tokenizer::BpeTokenizer;
use serde::{Deserialize, Serialize};

use crate::distributed::{
    DistributedConfig, DistributedContext, DistributedRuntime, ResolvedDistributedConfig,
    resolve_runtime,
};
use crate::trainer::Trainer;
use crate::vision_projector::{self, VisionTrainingConfig};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
/// One progressive context-length training stage.
pub struct ContextScheduleStage {
    /// Sequence length used when rebuilding train and validation loaders.
    pub max_seq_len: usize,
    /// Optimizer step at which this stage finishes.
    pub until_step: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
/// Complete TOML configuration for a training run.
pub struct TrainingRunConfig {
    /// Path to plaintext training data.
    pub dataset_path: PathBuf,
    /// Optional existing tokenizer JSON path.
    pub tokenizer_path: Option<PathBuf>,
    /// Optional path where a newly trained tokenizer is saved.
    pub tokenizer_save_path: Option<PathBuf>,
    /// Target tokenizer vocabulary size when training a tokenizer.
    pub vocab_size: usize,
    /// Fraction of records reserved for validation.
    pub validation_split: f64,
    /// Whether to shuffle training batches each epoch.
    pub shuffle: bool,
    /// Whether to resume from the latest checkpoint.
    pub resume: bool,
    /// Device selector string such as `cpu` or `cuda:0`.
    pub device: String,
    /// Dtype selector string such as `f32`, `f16`, or `bf16`.
    pub dtype: String,
    /// Model architecture configuration.
    pub model: ModelConfig,
    /// Optimizer and schedule configuration.
    pub train: TrainConfig,
    /// Optional single-node distributed data-parallel configuration.
    pub distributed: Option<DistributedConfig>,
    /// Optional progressive sequence-length schedule for long-context continuation.
    pub context_schedule: Vec<ContextScheduleStage>,
    /// Optional vision training mode and data configuration.
    pub vision: Option<VisionTrainingConfig>,
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
            dtype: "f32".to_string(),
            model: ModelConfig::tiny(),
            train: TrainConfig::default(),
            distributed: None,
            context_schedule: Vec::new(),
            vision: None,
        }
    }
}

impl TrainingRunConfig {
    /// Load a training configuration from TOML.
    pub fn from_toml(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        toml::from_str(&content).map_err(|err| {
            AarambhError::Config(format!(
                "failed to parse training config {}: {err}",
                path.as_ref().display()
            ))
        })
    }

    /// Parse the configured device selector.
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

    /// Parse the configured dtype selector.
    pub fn dtype(&self) -> Result<AarambhDType> {
        self.dtype.parse()
    }

    /// Parse dtype and validate that it is compatible with `device`.
    pub fn dtype_for_device(&self, device: &Device) -> Result<AarambhDType> {
        let dtype = self.dtype()?;
        if device.is_cpu() && dtype != AarambhDType::F32 {
            return Err(AarambhError::Config(format!(
                "dtype {dtype} requires a GPU device; use dtype = \"f32\" for CPU runs"
            )));
        }
        Ok(dtype)
    }

    /// Validate required paths and numeric ranges.
    pub fn validate(&self) -> Result<()> {
        if self.vision.is_none() && self.dataset_path.as_os_str().is_empty() {
            return Err(AarambhError::Config("dataset_path is required".into()));
        }
        if let Some(vision) = &self.vision {
            vision.validate()?;
        }
        if !(0.0..1.0).contains(&self.validation_split) {
            return Err(AarambhError::Config(
                "validation_split must be in [0, 1)".into(),
            ));
        }
        if self.vocab_size == 0 {
            return Err(AarambhError::Config("vocab_size must be non-zero".into()));
        }
        let device = self.device()?;
        self.dtype_for_device(&device)?;
        if let Some(distributed) = &self.distributed {
            distributed.validate()?;
        }
        self.validate_context_schedule()?;
        Ok(())
    }

    fn validate_context_schedule(&self) -> Result<()> {
        if self.context_schedule.is_empty() {
            return Ok(());
        }
        let mut prev_step = 0usize;
        for stage in &self.context_schedule {
            if stage.max_seq_len == 0 {
                return Err(AarambhError::Config(
                    "context_schedule.max_seq_len must be non-zero".into(),
                ));
            }
            if stage.max_seq_len > self.model.max_seq_len {
                return Err(AarambhError::Config(format!(
                    "context_schedule.max_seq_len {} exceeds model.max_seq_len {}",
                    stage.max_seq_len, self.model.max_seq_len
                )));
            }
            if stage.until_step <= prev_step {
                return Err(AarambhError::Config(
                    "context_schedule.until_step values must be strictly increasing".into(),
                ));
            }
            prev_step = stage.until_step;
        }
        let final_step = self
            .context_schedule
            .last()
            .map(|stage| stage.until_step)
            .unwrap_or(0);
        if final_step != self.train.max_steps {
            return Err(AarambhError::Config(format!(
                "final context_schedule.until_step {final_step} must equal train.max_steps {}",
                self.train.max_steps
            )));
        }
        Ok(())
    }
}

/// Load a TOML config and execute the training run.
pub fn run_training_from_config(path: impl AsRef<Path>) -> Result<()> {
    let config = TrainingRunConfig::from_toml(path)?;
    config.validate()?;
    if let Some(vision) = &config.vision
        && vision.mode == "projector_pretrain"
    {
        return vision_projector::run_projector_pretrain(&config);
    }

    let runtime = resolve_runtime(config.distributed.as_ref())?;
    if let DistributedRuntime::NonParticipant {
        rank,
        world_size,
        reason,
    } = &runtime
    {
        println!(
            "distributed fallback: rank {rank}/{world_size} exiting because rank 0 will run single-process training ({reason})"
        );
        return Ok(());
    }

    let mut device = config.device()?;
    let distributed_config = match &runtime {
        DistributedRuntime::Active(distributed) => {
            device = Device::Cuda(distributed.local_rank);
            Some(distributed.clone())
        }
        DistributedRuntime::SingleProcessFallback { reason, .. } => {
            if matches!(device, Device::Cuda(_)) {
                device = Device::Cuda(0);
            }
            println!("distributed fallback: {reason}; running single-process training on rank 0");
            None
        }
        DistributedRuntime::Disabled | DistributedRuntime::NonParticipant { .. } => None,
    };
    let is_rank0 = runtime.is_rank0();
    let dtype = config.dtype_for_device(&device)?.to_candle();
    let candle_device = device.to_candle()?;
    let distributed_context = distributed_config
        .clone()
        .map(|distributed| DistributedContext::init(distributed, &candle_device))
        .transpose()?;
    if let Some(distributed) = &distributed_config {
        if is_rank0 {
            println!(
                "distributed training: backend={:?} world_size={} rank={} local_rank={} dtype={dtype:?}",
                distributed.backend,
                distributed.world_size,
                distributed.rank,
                distributed.local_rank
            );
        }
    } else if is_rank0 {
        println!("training run: device={device:?} dtype={dtype:?}");
    }
    let tokenizer = prepare_tokenizer(
        &config,
        is_rank0,
        distributed_config
            .as_ref()
            .map(|distributed| distributed.init_timeout_secs)
            .unwrap_or(120),
    )?;
    let mut model_config = config.model.clone();
    model_config.vocab_size = tokenizer.vocab_size();

    let (train_dataset, val_dataset) = load_datasets(&config)?;
    let initial_seq_len = config
        .context_schedule
        .first()
        .map(|stage| stage.max_seq_len)
        .unwrap_or(model_config.max_seq_len);
    let (train_loader, val_loader) = build_loaders(
        &train_dataset,
        val_dataset.as_ref(),
        &tokenizer,
        &config,
        initial_seq_len,
        device.clone(),
        distributed_config.as_ref(),
    );

    let mut trainer = Trainer::new_with_distributed(
        model_config,
        config.train.clone(),
        train_loader,
        val_loader,
        candle_device,
        dtype,
        distributed_context,
    )?;
    if config.resume && trainer.load_latest_checkpoint()? && trainer.is_rank0() {
        println!("resumed checkpoint at step={}", trainer.state().step);
    }
    if config.context_schedule.is_empty() {
        trainer.train()
    } else {
        for stage in &config.context_schedule {
            if trainer.state().step >= stage.until_step {
                continue;
            }
            let (train_loader, val_loader) = build_loaders(
                &train_dataset,
                val_dataset.as_ref(),
                &tokenizer,
                &config,
                stage.max_seq_len,
                device.clone(),
                distributed_config.as_ref(),
            );
            trainer.replace_loaders(train_loader, val_loader);
            if trainer.is_rank0() {
                println!(
                    "context stage: max_seq_len={} until_step={}",
                    stage.max_seq_len, stage.until_step
                );
            }
            trainer.train_until(stage.until_step)?;
        }
        trainer.save_checkpoint()
    }
}

fn prepare_tokenizer(
    config: &TrainingRunConfig,
    is_rank0: bool,
    wait_timeout_secs: u64,
) -> Result<BpeTokenizer> {
    if let Some(path) = &config.tokenizer_path {
        let tokenizer = BpeTokenizer::from_pretrained(path)?;
        tokenizer.validate_special_tokens()?;
        return Ok(tokenizer);
    }

    fs::create_dir_all(&config.train.checkpoint_dir)?;
    let save_path = config
        .tokenizer_save_path
        .clone()
        .unwrap_or_else(|| config.train.checkpoint_dir.join("tokenizer.json"));
    if let Some(tokenizer) = load_valid_tokenizer(&save_path)? {
        return Ok(tokenizer);
    }

    if !is_rank0 {
        return wait_for_tokenizer(&save_path, wait_timeout_secs);
    }

    let tokenizer = BpeTokenizer::train(&config.dataset_path, config.vocab_size)?;
    tokenizer.validate_special_tokens()?;
    tokenizer.save_pretrained(save_path)?;
    Ok(tokenizer)
}

fn load_valid_tokenizer(path: &Path) -> Result<Option<BpeTokenizer>> {
    if !path.exists() {
        return Ok(None);
    }
    let tokenizer = BpeTokenizer::from_pretrained(path)?;
    if tokenizer.validate_special_tokens().is_ok() {
        Ok(Some(tokenizer))
    } else {
        Ok(None)
    }
}

fn wait_for_tokenizer(path: &Path, wait_timeout_secs: u64) -> Result<BpeTokenizer> {
    let deadline = Instant::now() + Duration::from_secs(wait_timeout_secs.max(1));
    loop {
        if path.exists() {
            match load_valid_tokenizer(path) {
                Ok(Some(tokenizer)) => return Ok(tokenizer),
                Ok(None) => {}
                Err(_) if Instant::now() < deadline => {}
                Err(err) => return Err(err),
            }
        }
        if Instant::now() >= deadline {
            return Err(AarambhError::Config(format!(
                "timed out waiting for rank 0 tokenizer at {}",
                path.display()
            )));
        }
        thread::sleep(Duration::from_millis(100));
    }
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

fn build_loaders(
    train_dataset: &PlaintextDataset,
    val_dataset: Option<&PlaintextDataset>,
    tokenizer: &BpeTokenizer,
    config: &TrainingRunConfig,
    max_seq_len: usize,
    device: Device,
    distributed: Option<&ResolvedDistributedConfig>,
) -> (DataLoader, Option<DataLoader>) {
    let train_loader = if let Some(distributed) = distributed {
        DataLoader::new_sharded(
            train_dataset,
            tokenizer,
            config.train.batch_size,
            max_seq_len,
            config.shuffle,
            device.clone(),
            DataShard {
                rank: distributed.rank,
                count: distributed.world_size,
                seed: config.train.seed.saturating_add(distributed.rank as u64),
            },
        )
    } else {
        DataLoader::new(
            train_dataset,
            tokenizer,
            config.train.batch_size,
            max_seq_len,
            config.shuffle,
            device.clone(),
        )
    };
    let val_loader = val_dataset.and_then(|dataset| {
        if distributed.is_some_and(|distributed| !distributed.is_rank0()) {
            return None;
        }
        Some(DataLoader::new_with_seed(
            dataset,
            tokenizer,
            config.train.batch_size,
            max_seq_len,
            false,
            device,
            config.train.seed,
        ))
    });
    (train_loader, val_loader)
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

    #[test]
    fn parses_cuda_alias_as_device_zero() {
        let config = TrainingRunConfig {
            device: "cuda".into(),
            ..TrainingRunConfig::default()
        };
        assert_eq!(config.device().unwrap(), Device::Cuda(0));
    }

    #[test]
    fn parses_explicit_cuda_device_index() {
        let config = TrainingRunConfig {
            device: "cuda:1".into(),
            ..TrainingRunConfig::default()
        };
        assert_eq!(config.device().unwrap(), Device::Cuda(1));
    }

    #[test]
    fn parses_mixed_dtype_alias() {
        let config = TrainingRunConfig {
            dtype: "mixed".into(),
            ..TrainingRunConfig::default()
        };
        assert_eq!(config.dtype().unwrap(), AarambhDType::BF16);
    }

    #[test]
    fn cpu_rejects_non_f32_dtype() {
        let config = TrainingRunConfig {
            dataset_path: "data.txt".into(),
            dtype: "bf16".into(),
            ..TrainingRunConfig::default()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("requires a GPU device"), "{err}");
    }

    #[test]
    fn context_schedule_must_end_at_max_steps() {
        let config = TrainingRunConfig {
            dataset_path: "data.txt".into(),
            context_schedule: vec![ContextScheduleStage {
                max_seq_len: 4,
                until_step: 1,
            }],
            ..TrainingRunConfig::default()
        };
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("train.max_steps"), "{err}");
    }
}
