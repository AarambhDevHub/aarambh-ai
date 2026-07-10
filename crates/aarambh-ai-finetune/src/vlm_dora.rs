use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use aarambh_ai_core::{AarambhError, Device, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_ai_tokenizer::{BpeTokenizer, IMAGE, IMAGE_END, IMAGE_ID};
use aarambh_ai_train::optim::clip_gradients;
use aarambh_ai_train::{
    AdamW, AdamWConfig, CosineScheduleWithWarmup, GradMap, TrainState, VisionTrainingConfig,
    cross_entropy_loss,
};
use aarambh_ai_vision::{
    ClipVisionEncoder, ImagePreprocessor, ProjectorConfig, VisionEncoderConfig,
    VisionPreprocessConfig, VisionProjector, VqaExample, interleave_image_tokens, load_vqa_jsonl,
};
use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::Serialize;

use crate::adapter::{AdapterMetadata, AdapterMethod, save_adapter};
use crate::dora::DoraAarambhModel;
use crate::lora::LoraConfig;
use crate::sft::ChatTemplate;

/// Configuration for one vision-language DoRA instruction-tuning run.
#[derive(Debug, Clone)]
pub struct VlmDoraRunConfig {
    /// Base model configuration.
    pub model_config: ModelConfig,
    /// Training hyperparameters.
    pub train_config: TrainConfig,
    /// Frozen base model path.
    pub base_model_path: PathBuf,
    /// Tokenizer JSON path. Must include Phase 19 vision tokens.
    pub tokenizer_path: PathBuf,
    /// VQA instruction JSONL data path.
    pub data_path: PathBuf,
    /// Output adapter directory.
    pub output_dir: PathBuf,
    /// DoRA adapter configuration.
    pub lora_config: LoraConfig,
    /// Logical training device.
    pub device: Device,
    /// Candle dtype used for model, encoder, and projector weights.
    pub dtype: DType,
    /// Whether to use QDoRA quantized base linear weights.
    pub qdora: bool,
    /// Whether to shuffle VQA examples each epoch.
    pub shuffle: bool,
    /// Phase 19 vision paths and dimensions.
    pub vision: VisionTrainingConfig,
    /// Phase 19 projector checkpoint to initialize from.
    pub projector_path: PathBuf,
    /// Whether projector parameters are updated alongside DoRA adapters.
    pub train_projector: bool,
}

/// Metrics emitted by a VLM DoRA optimizer step.
#[derive(Debug, Clone)]
pub struct VlmDoraMetrics {
    /// Current optimizer step.
    pub step: usize,
    /// Most recent micro-batch loss.
    pub loss: f64,
    /// Exponential of the loss.
    pub perplexity: f64,
    /// Learning rate.
    pub lr: f64,
    /// DoRA adapter gradient norm.
    pub dora_grad_norm: f64,
    /// Projector gradient norm when projector training is enabled.
    pub projector_grad_norm: Option<f64>,
    /// Examples processed per second since the last log.
    pub samples_per_second: f64,
}

/// Trainer for vision-language instruction tuning with a frozen encoder and DoRA LLM adapters.
pub struct VlmDoraTrainer {
    model: DoraAarambhModel,
    dora_varmap: VarMap,
    dora_optimizer: AdamW,
    projector: VisionProjector,
    projector_varmap: VarMap,
    projector_optimizer: Option<AdamW>,
    encoder: ClipVisionEncoder,
    preprocess: ImagePreprocessor,
    tokenizer: BpeTokenizer,
    train_config: TrainConfig,
    schedule: CosineScheduleWithWarmup,
    output_dir: PathBuf,
    metadata: AdapterMetadata,
    vlm_metadata: VlmArtifactsMetadata,
    examples: Vec<VqaExample>,
    shuffle: bool,
    rng: StdRng,
    state: TrainState,
    dora_pending_grads: GradMap,
    projector_pending_grads: GradMap,
    device: candle_core::Device,
    last_loss: Option<f64>,
    samples_since_log: usize,
    last_log_at: Instant,
}

impl VlmDoraTrainer {
    /// Create a VLM DoRA trainer from loaded model, vision, and data components.
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: DoraAarambhModel,
        dora_varmap: VarMap,
        projector: VisionProjector,
        projector_varmap: VarMap,
        encoder: ClipVisionEncoder,
        preprocess: ImagePreprocessor,
        tokenizer: BpeTokenizer,
        train_config: TrainConfig,
        output_dir: impl Into<PathBuf>,
        metadata: AdapterMetadata,
        vlm_metadata: VlmArtifactsMetadata,
        examples: Vec<VqaExample>,
        shuffle: bool,
        train_projector: bool,
        device: candle_core::Device,
    ) -> Result<Self> {
        if train_config.batch_size == 0 || train_config.grad_accum_steps == 0 {
            return Err(AarambhError::Config(
                "batch_size and grad_accum_steps must be greater than zero".into(),
            ));
        }
        if train_config.max_steps == 0 {
            return Err(AarambhError::Config(
                "max_steps must be greater than zero".into(),
            ));
        }
        if examples.is_empty() {
            return Err(AarambhError::Config(
                "VLM DoRA trainer needs at least one VQA example".into(),
            ));
        }
        let dora_optimizer = AdamW::from_varmap(&dora_varmap, AdamWConfig::from(&train_config))?;
        if dora_optimizer.parameters().is_empty() {
            return Err(AarambhError::Config(
                "VLM DoRA target_modules produced zero trainable tensors".into(),
            ));
        }
        let projector_optimizer = if train_projector {
            let optimizer =
                AdamW::from_varmap(&projector_varmap, AdamWConfig::from(&train_config))?;
            if optimizer.parameters().is_empty() {
                return Err(AarambhError::Config(
                    "projector training requested but projector has zero trainable tensors".into(),
                ));
            }
            Some(optimizer)
        } else {
            None
        };

        let seed = train_config.seed;
        Ok(Self {
            model,
            dora_varmap,
            dora_optimizer,
            projector,
            projector_varmap,
            projector_optimizer,
            encoder,
            preprocess,
            tokenizer,
            schedule: CosineScheduleWithWarmup::from_train_config(&train_config),
            train_config,
            output_dir: output_dir.into(),
            metadata,
            vlm_metadata,
            examples,
            shuffle,
            rng: StdRng::seed_from_u64(seed),
            state: TrainState::default(),
            dora_pending_grads: GradMap::new(),
            projector_pending_grads: GradMap::new(),
            device,
            last_loss: None,
            samples_since_log: 0,
            last_log_at: Instant::now(),
        })
    }

    /// Return the training state.
    pub fn state(&self) -> &TrainState {
        &self.state
    }

    /// Run the full VLM DoRA training loop and save final artifacts.
    pub fn train(&mut self) -> Result<()> {
        let examples_per_step = self.train_config.batch_size * self.train_config.grad_accum_steps;
        let mut example_idx = 0usize;
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            if example_idx == 0 && self.shuffle {
                self.examples.shuffle(&mut self.rng);
            }
            if example_idx >= self.examples.len() {
                example_idx = 0;
                self.state.epoch += 1;
                continue;
            }
            let loss = self.example_loss(&self.examples[example_idx])?;
            example_idx += 1;
            let loss_value = loss.to_scalar::<f32>()? as f64;
            if !loss_value.is_finite() {
                return Err(AarambhError::Config(format!(
                    "non-finite VLM DoRA loss: {loss_value}"
                )));
            }
            let scaled = loss.affine(1.0 / examples_per_step as f64, 0.0)?;
            let grads = scaled.backward()?;
            self.accumulate_gradients(&grads)?;
            self.state.micro_step += 1;
            self.state.train_loss = Some(loss_value);
            self.last_loss = Some(loss_value);
            self.samples_since_log += 1;
            if self.state.micro_step.is_multiple_of(examples_per_step) {
                self.optimizer_step()?;
            }
        }
        if !self.dora_pending_grads.is_empty() && self.state.step < self.train_config.max_steps {
            self.optimizer_step()?;
        }
        self.save_final()
    }

    fn example_loss(&self, example: &VqaExample) -> Result<Tensor> {
        let image_path = resolve_image_path(&self.vlm_metadata.image_root, &example.image_path);
        let image = self
            .preprocess
            .preprocess_path(&image_path, &self.device)?
            .unsqueeze(0)?;
        let patch_tokens = self.encoder.forward(&image)?.detach();
        let projected = self.projector.forward(&patch_tokens)?;
        let image_tokens = projected.dims()[1];
        let (text_tokens, target_start_idx) = self.vlm_tokens(example, image_tokens)?;
        let text = Tensor::from_vec(text_tokens.clone(), (1, text_tokens.len()), &self.device)?;
        let text_embeddings = self.model.embed_tokens(&text)?.detach();
        let fused = interleave_image_tokens(&text_tokens, &text_embeddings, &projected, IMAGE_ID)?;
        let logits = self.model.forward_embeddings_train(&fused)?;
        let (labels, mask) =
            vlm_labels_and_mask(&text_tokens, target_start_idx, image_tokens, IMAGE_ID)?;
        let seq_len = labels.len();
        let labels = Tensor::from_vec(labels, (1, seq_len), &self.device)?;
        let mask = Tensor::from_vec(mask, (1, seq_len), &self.device)?;
        cross_entropy_loss(&logits, &labels, &mask)
    }

    fn vlm_tokens(&self, example: &VqaExample, image_tokens: usize) -> Result<(Vec<u32>, usize)> {
        if image_tokens == 0 || image_tokens > self.model.config().max_seq_len {
            return Err(AarambhError::Shape(format!(
                "image token count {image_tokens} is invalid for max_seq_len {}",
                self.model.config().max_seq_len
            )));
        }
        let template = ChatTemplate;
        let prefix = format!(
            "{IMAGE}{IMAGE_END}\n{}",
            template.prefix(&example.question, None)
        );
        let target = match &example.thinking {
            Some(thinking) => template.thinking_target(thinking, &example.answer),
            None => template.target(&example.answer),
        };
        let prefix_ids = self.tokenizer.encode(&prefix)?;
        let mut target_ids = self.tokenizer.encode(&target)?;
        if !prefix_ids.contains(&IMAGE_ID) {
            return Err(AarambhError::Tokenizer(
                "VLM prefix did not encode the image placeholder token".into(),
            ));
        }
        if target_ids.is_empty() {
            return Err(AarambhError::Config(
                "VLM target encoded to zero tokens".into(),
            ));
        }

        let max_text_tokens = self.model.config().max_seq_len + 1 - image_tokens;
        if prefix_ids.len() >= max_text_tokens {
            return Err(AarambhError::Shape(format!(
                "VLM prompt has {} text tokens plus {image_tokens} image tokens, exceeding max_seq_len {}",
                prefix_ids.len(),
                self.model.config().max_seq_len
            )));
        }
        let keep_target = max_text_tokens - prefix_ids.len();
        if target_ids.len() > keep_target {
            target_ids.truncate(keep_target);
        }
        let target_start_idx = prefix_ids.len();
        let mut tokens = prefix_ids;
        tokens.extend(target_ids);
        if tokens.len() < 2 {
            return Err(AarambhError::Config(
                "VLM sequence must contain at least two tokens".into(),
            ));
        }
        Ok((tokens, target_start_idx))
    }

    fn accumulate_gradients(&mut self, grads: &GradStore) -> Result<()> {
        accumulate_for_optimizer(
            grads,
            &self.dora_optimizer,
            &mut self.dora_pending_grads,
            "VLM DoRA",
        )?;
        if let Some(projector_optimizer) = &self.projector_optimizer {
            accumulate_for_optimizer(
                grads,
                projector_optimizer,
                &mut self.projector_pending_grads,
                "VLM projector",
            )?;
        }
        Ok(())
    }

    fn optimizer_step(&mut self) -> Result<()> {
        let lr = self.schedule.lr_at_step(self.state.step);
        let dora_grad_norm = clip_gradients(
            &mut self.dora_pending_grads,
            self.train_config.clip_grad_norm,
        )?;
        self.dora_optimizer.step(&self.dora_pending_grads, lr)?;
        self.dora_pending_grads.clear();

        let projector_grad_norm = if let Some(projector_optimizer) = &mut self.projector_optimizer {
            let norm = clip_gradients(
                &mut self.projector_pending_grads,
                self.train_config.clip_grad_norm,
            )?;
            projector_optimizer.step(&self.projector_pending_grads, lr)?;
            self.projector_pending_grads.clear();
            Some(norm)
        } else {
            None
        };

        self.state.step += 1;
        let metrics = VlmDoraMetrics {
            step: self.state.step,
            loss: self.last_loss.unwrap_or(0.0),
            perplexity: self.last_loss.unwrap_or(0.0).exp(),
            lr,
            dora_grad_norm,
            projector_grad_norm,
            samples_per_second: self.samples_per_second_since_last_log(),
        };
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&self, metrics: &VlmDoraMetrics) -> Result<()> {
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let projector = metrics
                .projector_grad_norm
                .map(|value| format!(" projector_grad_norm={value:.4}"))
                .unwrap_or_default();
            println!(
                "vlm_dora step={} loss={:.4} ppl={:.2} lr={:.6} dora_grad_norm={:.4}{} samples/s={:.2}",
                metrics.step,
                metrics.loss,
                metrics.perplexity,
                metrics.lr,
                metrics.dora_grad_norm,
                projector,
                metrics.samples_per_second
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

    fn samples_per_second_since_last_log(&mut self) -> f64 {
        let elapsed = self.last_log_at.elapsed().as_secs_f64();
        let samples = self.samples_since_log;
        self.samples_since_log = 0;
        self.last_log_at = Instant::now();
        if elapsed > 0.0 {
            samples as f64 / elapsed
        } else {
            0.0
        }
    }

    fn save_final(&self) -> Result<()> {
        save_vlm_artifacts(
            &self.dora_varmap,
            &self.projector_varmap,
            &self.metadata,
            &self.vlm_metadata,
            &self.state,
            &self.output_dir,
        )
    }

    fn save_step(&self) -> Result<()> {
        let dir = self
            .output_dir
            .join("checkpoints")
            .join(format!("step_{:06}", self.state.step));
        save_vlm_artifacts(
            &self.dora_varmap,
            &self.projector_varmap,
            &self.metadata,
            &self.vlm_metadata,
            &self.state,
            dir,
        )
    }
}

/// Build and run a VLM DoRA trainer from a run configuration.
pub fn run_vlm_dora_from_config(config: VlmDoraRunConfig) -> Result<()> {
    config.lora_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_vision_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();
    if model_config.moe.is_some() {
        return Err(AarambhError::Config(
            "VLM DoRA training for MoE models is not supported in Phase 22; train the MoE base model directly or use a dense config".into(),
        ));
    }

    let base = aarambh_ai_weights::load_any_model_with_dtype(
        &config.base_model_path,
        &model_config,
        &candle_device,
        config.dtype,
    )?;
    let base_tensors = base.named_tensors();
    drop(base);

    let (model, dora_varmap) = DoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        config.qdora,
        &candle_device,
    )?;
    eprintln!(
        "vlm adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let encoder_config = VisionEncoderConfig::from_json(&config.vision.clip_config_path)?;
    let encoder = ClipVisionEncoder::load_pretrained(
        &config.vision.clip_weights_path,
        encoder_config.clone(),
        &candle_device,
        config.dtype,
    )?;
    let preprocess = ImagePreprocessor::new(VisionPreprocessConfig {
        image_size: encoder_config.image_size,
        ..VisionPreprocessConfig::default()
    })?;
    let projector_varmap = VarMap::new();
    let projector_vb = VarBuilder::from_varmap(&projector_varmap, config.dtype, &candle_device);
    let projector_config = ProjectorConfig {
        vit_d_model: encoder_config.vit_d_model,
        llm_d_model: model_config.hidden_dim,
        hidden_mult: config.vision.projector_hidden_mult,
    };
    let projector = VisionProjector::new(projector_config, projector_vb)?;
    let mut projector_varmap = projector_varmap;
    projector_varmap.load(&config.projector_path)?;

    let examples = load_vqa_jsonl(&config.data_path, config.vision.max_samples)?;
    let metadata = AdapterMetadata::new_with_method(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        config.qdora,
        AdapterMethod::Dora,
    );
    let vlm_metadata = VlmArtifactsMetadata {
        format_version: 1,
        projector_path: "projector.safetensors".into(),
        train_projector: config.train_projector,
        base_projector_path: config.projector_path.display().to_string(),
        clip_config_path: config.vision.clip_config_path.display().to_string(),
        clip_weights_path: config.vision.clip_weights_path.display().to_string(),
        image_root: config.vision.image_root.clone(),
    };
    let mut trainer = VlmDoraTrainer::new(
        model,
        dora_varmap,
        projector,
        projector_varmap,
        encoder,
        preprocess,
        tokenizer,
        config.train_config,
        config.output_dir,
        metadata,
        vlm_metadata,
        examples,
        config.shuffle,
        config.train_projector,
        candle_device,
    )?;
    trainer.train()
}

#[derive(Debug, Clone, Serialize)]
struct VlmArtifactsMetadata {
    format_version: u32,
    projector_path: String,
    train_projector: bool,
    base_projector_path: String,
    clip_config_path: String,
    clip_weights_path: String,
    image_root: PathBuf,
}

fn accumulate_for_optimizer(
    grads: &GradStore,
    optimizer: &AdamW,
    pending: &mut GradMap,
    label: &str,
) -> Result<()> {
    let mut updates = Vec::new();
    for param in optimizer.parameters() {
        let Some(grad) = grads.get(param.tensor()) else {
            continue;
        };
        let grad = grad.detach();
        let next = match pending.get(param.name()) {
            Some(existing) => ((existing + &grad)?).detach(),
            None => grad,
        };
        updates.push((param.name().to_string(), next));
    }
    if updates.is_empty() {
        return Err(AarambhError::Config(format!(
            "{label} backward produced no trainable parameter gradients"
        )));
    }
    for (name, grad) in updates {
        pending.insert(name, grad);
    }
    Ok(())
}

fn save_vlm_artifacts(
    dora_varmap: &VarMap,
    projector_varmap: &VarMap,
    metadata: &AdapterMetadata,
    vlm_metadata: &VlmArtifactsMetadata,
    state: &TrainState,
    output_dir: impl AsRef<Path>,
) -> Result<()> {
    let output_dir = output_dir.as_ref();
    save_adapter(dora_varmap, metadata, output_dir)?;
    projector_varmap.save(output_dir.join("projector.safetensors"))?;
    write_json(output_dir.join("vlm_config.json"), vlm_metadata)?;
    write_json(output_dir.join("train_state.json"), state)?;
    Ok(())
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}

fn resolve_image_path(root: &Path, image_path: &Path) -> PathBuf {
    if image_path.is_absolute() {
        image_path.to_path_buf()
    } else {
        root.join(image_path)
    }
}

fn vlm_labels_and_mask(
    text_tokens: &[u32],
    target_start_idx: usize,
    image_token_count: usize,
    image_placeholder_id: u32,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut items = Vec::new();
    for (idx, token) in text_tokens.iter().enumerate() {
        if *token == image_placeholder_id {
            items.extend(std::iter::repeat_n((None, idx), image_token_count));
        } else {
            items.push((Some(*token), idx));
        }
    }
    let mut labels = Vec::with_capacity(items.len());
    let mut mask = Vec::with_capacity(items.len());
    for idx in 0..items.len() {
        match items.get(idx + 1).copied() {
            Some((Some(token), original_idx)) if original_idx >= target_start_idx => {
                labels.push(token);
                mask.push(1);
            }
            Some((Some(token), _)) => {
                labels.push(token);
                mask.push(0);
            }
            _ => {
                labels.push(0);
                mask.push(0);
            }
        }
    }
    if !mask.contains(&1) {
        return Err(AarambhError::Shape(
            "VLM loss mask has no supervised answer tokens".into(),
        ));
    }
    Ok((labels, mask))
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_ai_tokenizer::{ENDOFTEXT_ID, IMAGE_END_ID};

    #[test]
    fn vlm_loss_mask_zeros_image_and_question_tokens() {
        let text = vec![IMAGE_ID, IMAGE_END_ID, 10, 11, 12, ENDOFTEXT_ID];
        let (labels, mask) = vlm_labels_and_mask(&text, 4, 3, IMAGE_ID).unwrap();
        assert_eq!(labels.len(), 8);
        assert_eq!(mask, vec![0, 0, 0, 0, 0, 1, 1, 0]);
        assert_eq!(labels[5], 12);
        assert_eq!(labels[6], ENDOFTEXT_ID);
    }
}
