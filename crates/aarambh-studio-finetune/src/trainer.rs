use std::fs;
use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Device, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_studio_tokenizer::BpeTokenizer;
use aarambh_studio_train::optim::clip_gradients;
use aarambh_studio_train::{
    AdamW, AdamWConfig, CosineScheduleWithWarmup, GradMap, TrainState, cross_entropy_loss,
};
use aarambh_studio_weights::load_any_model;
use candle_core::backprop::GradStore;
use candle_core::{Device as CandleDevice, Tensor};
use candle_nn::VarMap;
use serde::Serialize;

use crate::adapter::{
    AdapterMetadata, AdapterMethod, load_adapter_metadata, load_adapter_weights, save_adapter,
};
use crate::dora::DoraAarambhModel;
use crate::lora::LoraConfig;
use crate::model::LoraAarambhModel;
use crate::sft::{SftDataLoader, SftDataset};
use crate::tool_sft::ToolSftDataset;

#[derive(Debug, Clone)]
/// Configuration for one SFT adapter training run.
pub struct SftRunConfig {
    /// Base model configuration.
    pub model_config: ModelConfig,
    /// Training hyperparameters.
    pub train_config: TrainConfig,
    /// Path to base safetensors or GGUF checkpoint.
    pub base_model_path: PathBuf,
    /// Path to tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Path to SFT JSONL data.
    pub data_path: PathBuf,
    /// Directory where adapter artifacts are saved.
    pub output_dir: PathBuf,
    /// LoRA adapter configuration.
    pub lora_config: LoraConfig,
    /// Logical training device.
    pub device: Device,
    /// Whether to quantize base linear weights for QLoRA.
    pub qlora: bool,
    /// Whether to shuffle examples each epoch.
    pub shuffle: bool,
}

#[derive(Debug, Clone)]
/// Metrics emitted by an SFT training step.
pub struct SftMetrics {
    /// Current optimizer step.
    pub step: usize,
    /// Batch loss.
    pub loss: f64,
    /// Exponential of loss.
    pub perplexity: f64,
    /// Learning rate.
    pub lr: f64,
    /// Gradient norm when an optimizer step occurred.
    pub grad_norm: Option<f64>,
    /// Whether the micro-step performed an optimizer update.
    pub did_optimizer_step: bool,
}

/// Model interface required by the shared SFT adapter trainer.
pub trait AdapterSftModel {
    /// Run the training forward path.
    fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor>;
    /// Return the number of trainable adapter parameters.
    fn adapter_param_count(&self) -> usize;
    /// Return the number of frozen base parameters.
    fn base_param_count(&self) -> usize;
    /// Return trainable adapter parameters divided by base parameters.
    fn trainable_ratio(&self) -> f64;
    /// Return normal model tensors with adapter weights merged.
    fn merged_tensors(&self) -> Result<std::collections::HashMap<String, Tensor>>;
}

impl AdapterSftModel for LoraAarambhModel {
    fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_train(token_ids)
    }

    fn adapter_param_count(&self) -> usize {
        self.adapter_param_count()
    }

    fn base_param_count(&self) -> usize {
        self.base_param_count()
    }

    fn trainable_ratio(&self) -> f64 {
        self.trainable_ratio()
    }

    fn merged_tensors(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        self.merged_tensors()
    }
}

impl AdapterSftModel for DoraAarambhModel {
    fn forward_train(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.forward_train(token_ids)
    }

    fn adapter_param_count(&self) -> usize {
        self.adapter_param_count()
    }

    fn base_param_count(&self) -> usize {
        self.base_param_count()
    }

    fn trainable_ratio(&self) -> f64 {
        self.trainable_ratio()
    }

    fn merged_tensors(&self) -> Result<std::collections::HashMap<String, Tensor>> {
        self.merged_tensors()
    }
}

/// Trainer for LoRA, QLoRA, DoRA, and QDoRA supervised fine-tuning.
pub struct AdapterSftTrainer<M: AdapterSftModel> {
    model: M,
    varmap: VarMap,
    optimizer: AdamW,
    schedule: CosineScheduleWithWarmup,
    train_loader: SftDataLoader,
    train_config: TrainConfig,
    output_dir: PathBuf,
    metadata: AdapterMetadata,
    state: TrainState,
    pending_grads: GradMap,
    last_loss: Option<f64>,
}

/// Trainer for LoRA/QLoRA supervised fine-tuning.
pub type SftTrainer = AdapterSftTrainer<LoraAarambhModel>;

/// Trainer for DoRA/QDoRA supervised fine-tuning.
pub type DoraTrainer = AdapterSftTrainer<DoraAarambhModel>;

impl<M: AdapterSftModel> AdapterSftTrainer<M> {
    /// Create an adapter SFT trainer.
    pub fn new(
        model: M,
        varmap: VarMap,
        train_loader: SftDataLoader,
        train_config: TrainConfig,
        output_dir: impl Into<PathBuf>,
        metadata: AdapterMetadata,
    ) -> Result<Self> {
        if train_config.grad_accum_steps == 0 {
            return Err(AarambhError::Config(
                "grad_accum_steps must be greater than zero".into(),
            ));
        }
        if train_config.max_steps == 0 {
            return Err(AarambhError::Config(
                "max_steps must be greater than zero".into(),
            ));
        }
        if train_loader.is_empty() {
            return Err(AarambhError::Config(
                "SFT dataloader has no full batches".into(),
            ));
        }
        let optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&train_config))?;
        if optimizer.parameters().is_empty() {
            return Err(AarambhError::Config(
                "adapter target_modules produced zero trainable tensors".into(),
            ));
        }
        let schedule = CosineScheduleWithWarmup::from_train_config(&train_config);
        Ok(Self {
            model,
            varmap,
            optimizer,
            schedule,
            train_loader,
            train_config,
            output_dir: output_dir.into(),
            metadata,
            state: TrainState::default(),
            pending_grads: GradMap::new(),
            last_loss: None,
        })
    }

    /// Return the adapter model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Return trainable adapter variables.
    pub fn varmap(&self) -> &VarMap {
        &self.varmap
    }

    /// Return training state.
    pub fn state(&self) -> &TrainState {
        &self.state
    }

    /// Run one SFT training micro-step.
    pub fn train_step(&mut self, batch: crate::sft::SftBatch) -> Result<SftMetrics> {
        let logits = self.model.forward_train(&batch.input_ids)?;
        let loss = cross_entropy_loss(&logits, &batch.labels, &batch.loss_mask)?;
        let loss_value = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(AarambhError::Config(format!(
                "non-finite SFT loss: {loss_value}"
            )));
        }

        let scaled_loss = loss.affine(1.0 / self.train_config.grad_accum_steps as f64, 0.0)?;
        let grads = scaled_loss.backward()?;
        self.accumulate_gradients(&grads)?;
        self.state.micro_step += 1;
        self.state.train_loss = Some(loss_value);
        self.last_loss = Some(loss_value);

        let should_step = self
            .state
            .micro_step
            .is_multiple_of(self.train_config.grad_accum_steps);
        if should_step {
            let (lr, grad_norm) = self.optimizer_step()?;
            Ok(SftMetrics {
                step: self.state.step,
                loss: loss_value,
                perplexity: loss_value.exp(),
                lr,
                grad_norm: Some(grad_norm),
                did_optimizer_step: true,
            })
        } else {
            Ok(SftMetrics {
                step: self.state.step,
                loss: loss_value,
                perplexity: loss_value.exp(),
                lr: self.schedule.lr_at_step(self.state.step),
                grad_norm: None,
                did_optimizer_step: false,
            })
        }
    }

    /// Train until the epoch or max-step boundary completes.
    pub fn train_epoch(&mut self) -> Result<()> {
        self.train_loader.reset();
        while self.state.step < self.train_config.max_steps {
            let Some(batch) = self.train_loader.next() else {
                break;
            };
            let metrics = self.train_step(batch?)?;
            if metrics.did_optimizer_step {
                self.after_optimizer_step(&metrics)?;
            }
        }
        self.flush_pending_step()?;
        self.state.epoch += 1;
        Ok(())
    }

    /// Run the full SFT training loop and save final adapter artifacts.
    pub fn train(&mut self) -> Result<()> {
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            self.train_epoch()?;
        }
        self.save_final()
    }

    /// Save final adapter artifacts.
    pub fn save_final(&self) -> Result<()> {
        save_adapter(&self.varmap, &self.metadata, &self.output_dir)?;
        write_json(self.output_dir.join("train_state.json"), &self.state)?;
        Ok(())
    }

    fn save_step(&self) -> Result<()> {
        let dir = self
            .output_dir
            .join("checkpoints")
            .join(format!("step_{:06}", self.state.step));
        save_adapter(&self.varmap, &self.metadata, &dir)?;
        write_json(dir.join("train_state.json"), &self.state)?;
        Ok(())
    }

    fn accumulate_gradients(&mut self, grads: &GradStore) -> Result<()> {
        let mut updates = Vec::new();
        for param in self.optimizer.parameters() {
            let Some(grad) = grads.get(param.tensor()) else {
                continue;
            };
            let grad = grad.detach();
            let next = match self.pending_grads.get(param.name()) {
                Some(existing) => ((existing + &grad)?).detach(),
                None => grad,
            };
            updates.push((param.name().to_string(), next));
        }

        if updates.is_empty() {
            return Err(AarambhError::Config(
                "SFT backward produced no adapter parameter gradients".into(),
            ));
        }
        for (name, grad) in updates {
            self.pending_grads.insert(name, grad);
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<(f64, f64)> {
        let lr = self.schedule.lr_at_step(self.state.step);
        let grad_norm = clip_gradients(&mut self.pending_grads, self.train_config.clip_grad_norm)?;
        self.optimizer.step(&self.pending_grads, lr)?;
        self.pending_grads.clear();
        self.state.step += 1;
        Ok((lr, grad_norm))
    }

    fn flush_pending_step(&mut self) -> Result<()> {
        if self.pending_grads.is_empty() || self.state.step >= self.train_config.max_steps {
            return Ok(());
        }
        let (lr, grad_norm) = self.optimizer_step()?;
        let loss = self.last_loss.unwrap_or(0.0);
        let metrics = SftMetrics {
            step: self.state.step,
            loss,
            perplexity: loss.exp(),
            lr,
            grad_norm: Some(grad_norm),
            did_optimizer_step: true,
        };
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&mut self, metrics: &SftMetrics) -> Result<()> {
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let grad_norm = metrics.grad_norm.unwrap_or(0.0);
            let method = adapter_method_label(self.metadata.method);
            println!(
                "{method} step={} loss={:.4} ppl={:.2} lr={:.6} grad_norm={:.4}",
                metrics.step, metrics.loss, metrics.perplexity, metrics.lr, grad_norm
            );
        }
        if self.train_config.save_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.save_every_n_steps)
        {
            self.save_step()?;
        }
        Ok(())
    }
}

/// Build and run an SFT trainer from a run configuration.
pub fn run_sft_from_config(config: SftRunConfig) -> Result<()> {
    config.lora_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    reject_moe_adapters(&model_config)?;

    let base = load_any_model(&config.base_model_path, &model_config, &candle_device)?;
    let base_tensors = base.named_tensors();
    drop(base);

    let (model, varmap) = LoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qlora,
        &candle_device,
    )?;
    eprintln!(
        "adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let dataset = SftDataset::from_jsonl(&config.data_path, &tokenizer, model_config.max_seq_len)?;
    let loader = SftDataLoader::new(
        &dataset,
        config.train_config.batch_size,
        config.shuffle,
        config.train_config.seed,
        config.device.clone(),
    )?;
    let metadata = AdapterMetadata::new(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qlora,
    );
    let mut trainer = SftTrainer::new(
        model,
        varmap,
        loader,
        config.train_config,
        config.output_dir,
        metadata,
    )?;
    trainer.train()
}

/// Build and run a LoRA or QLoRA tool-calling SFT trainer.
pub fn run_tool_sft_from_config(config: SftRunConfig) -> Result<()> {
    config.lora_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    reject_moe_adapters(&model_config)?;

    let base = load_any_model(&config.base_model_path, &model_config, &candle_device)?;
    let base_tensors = base.named_tensors();
    drop(base);
    let (model, varmap) = LoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qlora,
        &candle_device,
    )?;
    eprintln!(
        "tool adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let dataset =
        ToolSftDataset::from_jsonl(&config.data_path, &tokenizer, model_config.max_seq_len)?
            .into_inner();
    let loader = SftDataLoader::new(
        &dataset,
        config.train_config.batch_size,
        config.shuffle,
        config.train_config.seed,
        config.device.clone(),
    )?;
    let metadata = AdapterMetadata::new(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qlora,
    );
    let mut trainer = SftTrainer::new(
        model,
        varmap,
        loader,
        config.train_config,
        config.output_dir,
        metadata,
    )?;
    trainer.train()
}

/// Build and run a DoRA SFT trainer from a run configuration.
pub fn run_dora_from_config(config: SftRunConfig) -> Result<()> {
    config.lora_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    reject_moe_adapters(&model_config)?;

    let base = load_any_model(&config.base_model_path, &model_config, &candle_device)?;
    let base_tensors = base.named_tensors();
    drop(base);

    let (model, varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qlora,
        &candle_device,
    )?;
    eprintln!(
        "adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let dataset = SftDataset::from_jsonl(&config.data_path, &tokenizer, model_config.max_seq_len)?;
    let loader = SftDataLoader::new(
        &dataset,
        config.train_config.batch_size,
        config.shuffle,
        config.train_config.seed,
        config.device.clone(),
    )?;
    let metadata = AdapterMetadata::new_with_method(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qlora,
        AdapterMethod::Dora,
    );
    let mut trainer = DoraTrainer::new(
        model,
        varmap,
        loader,
        config.train_config,
        config.output_dir,
        metadata,
    )?;
    trainer.train()
}

/// Merge an adapter into base model weights by reading the adapter method.
pub fn merge_adapter_from_paths(
    model_config: &ModelConfig,
    base_model_path: impl AsRef<Path>,
    adapter_dir: impl AsRef<Path>,
    output: impl AsRef<Path>,
    device: &CandleDevice,
    method_override: Option<AdapterMethod>,
) -> Result<PathBuf> {
    let base_model_path = base_model_path.as_ref();
    let adapter_dir = adapter_dir.as_ref();
    let output = output.as_ref();
    let metadata = load_adapter_metadata(adapter_dir)?;
    let method = method_override.unwrap_or(metadata.method);
    match method {
        AdapterMethod::Lora => {
            merge_lora_from_paths(model_config, base_model_path, adapter_dir, output, device)
        }
        AdapterMethod::Dora => {
            merge_dora_from_paths(model_config, base_model_path, adapter_dir, output, device)
        }
    }
}

/// Merge a LoRA adapter into base model weights and save safetensors.
pub fn merge_lora_from_paths(
    model_config: &ModelConfig,
    base_model_path: impl AsRef<Path>,
    adapter_dir: impl AsRef<Path>,
    output: impl AsRef<Path>,
    device: &CandleDevice,
) -> Result<PathBuf> {
    let metadata = load_adapter_metadata(adapter_dir.as_ref())?;
    if metadata.method != AdapterMethod::Lora {
        return Err(AarambhError::Config(format!(
            "adapter method is {:?}, expected lora",
            metadata.method
        )));
    }
    let config = if model_config.vocab_size == metadata.model.vocab_size {
        model_config.clone()
    } else {
        metadata.model.clone()
    };
    reject_moe_adapters(&config)?;
    let base = load_any_model(base_model_path.as_ref(), &config, device)?;
    let tensors = base.named_tensors();
    drop(base);

    let (model, mut varmap) =
        LoraAarambhModel::from_tensors(&config, &tensors, &metadata.lora, false, device)?;
    load_adapter_weights(&mut varmap, adapter_dir.as_ref())?;
    save_merged_tensors(model.merged_tensors()?, output)
}

/// Merge a DoRA adapter into base model weights and save safetensors.
pub fn merge_dora_from_paths(
    model_config: &ModelConfig,
    base_model_path: impl AsRef<Path>,
    adapter_dir: impl AsRef<Path>,
    output: impl AsRef<Path>,
    device: &CandleDevice,
) -> Result<PathBuf> {
    let metadata = load_adapter_metadata(adapter_dir.as_ref())?;
    if metadata.method != AdapterMethod::Dora {
        return Err(AarambhError::Config(format!(
            "adapter method is {:?}, expected dora",
            metadata.method
        )));
    }
    let config = if model_config.vocab_size == metadata.model.vocab_size {
        model_config.clone()
    } else {
        metadata.model.clone()
    };
    reject_moe_adapters(&config)?;
    let base = load_any_model(base_model_path.as_ref(), &config, device)?;
    let tensors = base.named_tensors();
    drop(base);

    let (model, mut varmap) =
        DoraAarambhModel::from_tensors(&config, &tensors, &metadata.lora, false, device)?;
    load_adapter_weights(&mut varmap, adapter_dir.as_ref())?;
    save_merged_tensors(model.merged_tensors()?, output)
}

fn save_merged_tensors(
    merged: std::collections::HashMap<String, Tensor>,
    output: impl AsRef<Path>,
) -> Result<PathBuf> {
    let output = model_output_path(output);
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    candle_core::safetensors::save(&merged, &output)?;
    Ok(output)
}

fn model_output_path(output: impl AsRef<Path>) -> PathBuf {
    let output = output.as_ref();
    if output.extension().and_then(|ext| ext.to_str()) == Some("safetensors") {
        output.to_path_buf()
    } else {
        output.join("model.safetensors")
    }
}

fn reject_moe_adapters(model_config: &ModelConfig) -> Result<()> {
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "LoRA/DoRA adapter training for MoE models is not supported; train the MoE base model directly or use a dense config".into(),
        ));
    }
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}

fn adapter_method_label(method: AdapterMethod) -> &'static str {
    match method {
        AdapterMethod::Lora => "sft",
        AdapterMethod::Dora => "dora",
    }
}
