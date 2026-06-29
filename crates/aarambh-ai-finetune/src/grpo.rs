use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, Device, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_ai_model::AarambhModel;
use aarambh_ai_tokenizer::{BpeTokenizer, ENDOFTEXT_ID, THINK_END_ID, THINK_START_ID};
use aarambh_ai_train::optim::clip_gradients;
use aarambh_ai_train::{AdamW, AdamWConfig, CosineScheduleWithWarmup, GradMap, TrainState};
use aarambh_ai_weights::load_any_model;
use candle_core::backprop::GradStore;
use candle_core::{Device as CandleDevice, Tensor};
use candle_nn::VarMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::adapter::{AdapterMetadata, save_adapter};
use crate::lora::LoraConfig;
use crate::model::LoraAarambhModel;
use crate::sft::ChatTemplate;
use crate::verifier::{Verifier, VerifierKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GrpoConfig {
    pub group_size: usize,
    pub kl_coeff: f64,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub thinking: GrpoThinkingMode,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            kl_coeff: 0.01,
            max_new_tokens: 128,
            temperature: 0.8,
            top_p: Some(0.95),
            top_k: Some(50),
            thinking: GrpoThinkingMode::Low,
        }
    }
}

impl GrpoConfig {
    pub fn validate(&self) -> Result<()> {
        if self.group_size == 0 {
            return Err(AarambhError::Config(
                "GRPO group_size must be greater than zero".into(),
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(AarambhError::Config(
                "GRPO max_new_tokens must be greater than zero".into(),
            ));
        }
        if self.kl_coeff < 0.0 {
            return Err(AarambhError::Config(
                "GRPO kl_coeff must be non-negative".into(),
            ));
        }
        if self.temperature < 0.0 {
            return Err(AarambhError::Config(
                "GRPO temperature must be non-negative".into(),
            ));
        }
        if let Some(top_p) = self.top_p
            && !(0.0..=1.0).contains(&top_p)
        {
            return Err(AarambhError::Config("GRPO top_p must be in [0, 1]".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrpoThinkingMode {
    None,
    #[default]
    Low,
    Medium,
    High,
}

impl GrpoThinkingMode {
    pub fn budget(self) -> usize {
        match self {
            Self::None => 0,
            Self::Low => 256,
            Self::Medium => 1024,
            Self::High => 4096,
        }
    }

    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::None)
    }
}

impl std::str::FromStr for GrpoThinkingMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(format!(
                "unsupported thinking mode '{other}', expected none, low, medium, or high"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GrpoRunConfig {
    pub model_config: ModelConfig,
    pub train_config: TrainConfig,
    pub grpo_config: GrpoConfig,
    pub base_model_path: PathBuf,
    pub reference_model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub data_path: PathBuf,
    pub output_dir: PathBuf,
    pub lora_config: LoraConfig,
    pub verifier: VerifierKind,
    pub device: Device,
    pub shuffle: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrpoExample {
    pub prompt: String,
    pub ground_truth: String,
}

#[derive(Debug, Clone)]
pub struct GrpoDataset {
    examples: Vec<GrpoExample>,
}

impl GrpoDataset {
    pub fn from_jsonl(path: impl AsRef<Path>) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut examples = Vec::new();
        for (line_idx, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let raw: RawGrpoRecord = serde_json::from_str(line).map_err(|err| {
                AarambhError::Config(format!(
                    "invalid GRPO JSONL at line {}: {err}",
                    line_idx + 1
                ))
            })?;
            let prompt = raw.prompt.or(raw.question).ok_or_else(|| {
                AarambhError::Config(format!(
                    "GRPO JSONL line {} missing prompt or question",
                    line_idx + 1
                ))
            })?;
            let ground_truth = raw.ground_truth.or(raw.answer).ok_or_else(|| {
                AarambhError::Config(format!(
                    "GRPO JSONL line {} missing ground_truth or answer",
                    line_idx + 1
                ))
            })?;
            if prompt.trim().is_empty() || ground_truth.trim().is_empty() {
                return Err(AarambhError::Config(format!(
                    "GRPO JSONL line {} has an empty prompt or answer",
                    line_idx + 1
                )));
            }
            examples.push(GrpoExample {
                prompt,
                ground_truth,
            });
        }
        if examples.is_empty() {
            return Err(AarambhError::Config(format!(
                "GRPO dataset {} produced no usable examples",
                path.as_ref().display()
            )));
        }
        Ok(Self { examples })
    }

    pub fn from_examples(examples: Vec<GrpoExample>) -> Result<Self> {
        if examples.is_empty() {
            return Err(AarambhError::Config(
                "GRPO dataset must contain at least one example".into(),
            ));
        }
        Ok(Self { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

#[derive(Debug, Deserialize)]
struct RawGrpoRecord {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    question: Option<String>,
    #[serde(default)]
    ground_truth: Option<String>,
    #[serde(default)]
    answer: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RolloutFinish {
    Eos,
    MaxTokens,
    ContextLimit,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rollout {
    pub prompt_len: usize,
    pub completion_token_ids: Vec<u32>,
    pub completion_text: String,
    pub score: f32,
    pub advantage: f32,
    pub finish_reason: RolloutFinish,
}

#[derive(Debug, Clone)]
pub struct GrpoMetrics {
    pub step: usize,
    pub loss: f64,
    pub reward_mean: f64,
    pub reward_std: f64,
    pub advantage_mean: f64,
    pub kl: f64,
    pub lr: f64,
    pub grad_norm: Option<f64>,
    pub did_optimizer_step: bool,
}

pub struct GrpoTrainer {
    model: LoraAarambhModel,
    varmap: VarMap,
    reference: AarambhModel,
    tokenizer: BpeTokenizer,
    verifier: Box<dyn Verifier>,
    dataset: GrpoDataset,
    grpo_config: GrpoConfig,
    train_config: TrainConfig,
    output_dir: PathBuf,
    metadata: AdapterMetadata,
    save_metadata: GrpoSaveMetadata,
    optimizer: AdamW,
    schedule: CosineScheduleWithWarmup,
    state: TrainState,
    pending_grads: GradMap,
    last_loss: Option<f64>,
    order: Vec<usize>,
    order_pos: usize,
    shuffle: bool,
    rng: StdRng,
    device: CandleDevice,
}

impl GrpoTrainer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: LoraAarambhModel,
        varmap: VarMap,
        reference: AarambhModel,
        tokenizer: BpeTokenizer,
        verifier: Box<dyn Verifier>,
        dataset: GrpoDataset,
        grpo_config: GrpoConfig,
        train_config: TrainConfig,
        output_dir: impl Into<PathBuf>,
        metadata: AdapterMetadata,
        save_metadata: GrpoSaveMetadata,
        shuffle: bool,
        device: CandleDevice,
    ) -> Result<Self> {
        grpo_config.validate()?;
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
        if dataset.is_empty() {
            return Err(AarambhError::Config("GRPO dataset is empty".into()));
        }
        let optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&train_config))?;
        if optimizer.parameters().is_empty() {
            return Err(AarambhError::Config(
                "LoRA target_modules produced zero trainable adapter tensors".into(),
            ));
        }
        let schedule = CosineScheduleWithWarmup::from_train_config(&train_config);
        let mut trainer = Self {
            model,
            varmap,
            reference,
            tokenizer,
            verifier,
            order: (0..dataset.len()).collect(),
            dataset,
            grpo_config,
            train_config,
            output_dir: output_dir.into(),
            metadata,
            save_metadata,
            optimizer,
            schedule,
            state: TrainState::default(),
            pending_grads: GradMap::new(),
            last_loss: None,
            order_pos: 0,
            shuffle,
            rng: StdRng::seed_from_u64(42),
            device,
        };
        trainer.rng = StdRng::seed_from_u64(trainer.train_config.seed);
        trainer.reset_epoch_order();
        Ok(trainer)
    }

    pub fn model(&self) -> &LoraAarambhModel {
        &self.model
    }

    pub fn state(&self) -> &TrainState {
        &self.state
    }

    pub fn train(&mut self) -> Result<()> {
        while self.state.epoch < self.train_config.max_epochs
            && self.state.step < self.train_config.max_steps
        {
            self.train_epoch()?;
        }
        self.flush_pending_step()?;
        self.save_final()
    }

    pub fn train_epoch(&mut self) -> Result<()> {
        while self.state.step < self.train_config.max_steps {
            let example = match self.next_example() {
                Some(example) => example,
                None => break,
            };
            let metrics = self.train_step(&example)?;
            if metrics.did_optimizer_step {
                self.after_optimizer_step(&metrics)?;
            }
        }
        self.state.epoch += 1;
        self.reset_epoch_order();
        Ok(())
    }

    pub fn train_step(&mut self, example: &GrpoExample) -> Result<GrpoMetrics> {
        let mut rollouts = sample_group(
            &self.model,
            &self.tokenizer,
            example,
            &self.grpo_config,
            self.train_config.seed
                ^ ((self.state.step as u64) << 32)
                ^ self.state.micro_step as u64,
            &self.device,
        )?;
        for rollout in &mut rollouts {
            rollout.score = self
                .verifier
                .score(&rollout.completion_text, &example.ground_truth);
        }
        let scores = rollouts
            .iter()
            .map(|rollout| rollout.score)
            .collect::<Vec<_>>();
        let advantages = compute_advantages(&scores);
        for (rollout, advantage) in rollouts.iter_mut().zip(advantages.iter().copied()) {
            rollout.advantage = advantage;
        }

        let mut policy_log_probs = Vec::with_capacity(rollouts.len());
        let mut kl_terms = Vec::with_capacity(rollouts.len());
        for rollout in &rollouts {
            let (selected, kl) = replay_policy_terms(
                &self.model,
                &self.reference,
                &self.tokenizer,
                example,
                rollout,
                &self.device,
            )?;
            policy_log_probs.push(selected);
            kl_terms.push(kl);
        }
        let loss = grpo_loss_with_full_kl(
            &policy_log_probs,
            &kl_terms,
            &advantages,
            self.grpo_config.kl_coeff,
        )?;
        let loss_value = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(AarambhError::Config(format!(
                "non-finite GRPO loss: {loss_value}"
            )));
        }

        let scaled_loss = loss.affine(1.0 / self.train_config.grad_accum_steps as f64, 0.0)?;
        let grads = scaled_loss.backward()?;
        self.accumulate_gradients(&grads)?;
        self.state.micro_step += 1;
        self.state.train_loss = Some(loss_value);
        self.last_loss = Some(loss_value);

        let reward_mean = mean_f32(&scores) as f64;
        let reward_std = std_f32(&scores, reward_mean as f32) as f64;
        let advantage_mean = mean_f32(&advantages) as f64;
        let kl = mean_scalar_tensors(&kl_terms)?;

        let should_step = self
            .state
            .micro_step
            .is_multiple_of(self.train_config.grad_accum_steps);
        if should_step {
            let (lr, grad_norm) = self.optimizer_step()?;
            Ok(GrpoMetrics {
                step: self.state.step,
                loss: loss_value,
                reward_mean,
                reward_std,
                advantage_mean,
                kl,
                lr,
                grad_norm: Some(grad_norm),
                did_optimizer_step: true,
            })
        } else {
            Ok(GrpoMetrics {
                step: self.state.step,
                loss: loss_value,
                reward_mean,
                reward_std,
                advantage_mean,
                kl,
                lr: self.schedule.lr_at_step(self.state.step),
                grad_norm: None,
                did_optimizer_step: false,
            })
        }
    }

    pub fn save_final(&self) -> Result<()> {
        save_adapter(&self.varmap, &self.metadata, &self.output_dir)?;
        write_json(self.output_dir.join("train_state.json"), &self.state)?;
        write_json(
            self.output_dir.join("grpo_config.json"),
            &self.save_metadata,
        )?;
        Ok(())
    }

    fn save_step(&self) -> Result<()> {
        let dir = self
            .output_dir
            .join("checkpoints")
            .join(format!("step_{:06}", self.state.step));
        save_adapter(&self.varmap, &self.metadata, &dir)?;
        write_json(dir.join("train_state.json"), &self.state)?;
        write_json(dir.join("grpo_config.json"), &self.save_metadata)?;
        Ok(())
    }

    fn next_example(&mut self) -> Option<GrpoExample> {
        if self.order_pos >= self.order.len() {
            return None;
        }
        let idx = self.order[self.order_pos];
        self.order_pos += 1;
        Some(self.dataset.examples[idx].clone())
    }

    fn reset_epoch_order(&mut self) {
        self.order_pos = 0;
        if self.shuffle {
            self.order.shuffle(&mut self.rng);
        }
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
                "GRPO backward produced no LoRA parameter gradients".into(),
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
        let metrics = GrpoMetrics {
            step: self.state.step,
            loss: self.last_loss.unwrap_or(0.0),
            reward_mean: 0.0,
            reward_std: 0.0,
            advantage_mean: 0.0,
            kl: 0.0,
            lr,
            grad_norm: Some(grad_norm),
            did_optimizer_step: true,
        };
        self.after_optimizer_step(&metrics)
    }

    fn after_optimizer_step(&mut self, metrics: &GrpoMetrics) -> Result<()> {
        if self.train_config.log_every_n_steps > 0
            && metrics
                .step
                .is_multiple_of(self.train_config.log_every_n_steps)
        {
            let grad_norm = metrics.grad_norm.unwrap_or(0.0);
            println!(
                "grpo step={} loss={:.4} reward={:.3}±{:.3} adv={:.3} kl={:.5} lr={:.6} grad_norm={:.4}",
                metrics.step,
                metrics.loss,
                metrics.reward_mean,
                metrics.reward_std,
                metrics.advantage_mean,
                metrics.kl,
                metrics.lr,
                grad_norm
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoSaveMetadata {
    pub grpo: GrpoConfig,
    pub train: TrainConfig,
    pub verifier: VerifierKind,
    pub reference_model: String,
}

pub fn run_grpo_from_config(config: GrpoRunConfig) -> Result<()> {
    config.lora_config.validate()?;
    config.grpo_config.validate()?;
    let candle_device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    let mut model_config = config.model_config.clone();
    model_config.vocab_size = tokenizer.vocab_size();

    let base = load_any_model(&config.base_model_path, &model_config, &candle_device)?;
    let base_tensors = base.named_tensors();
    drop(base);
    let reference = load_any_model(&config.reference_model_path, &model_config, &candle_device)?;

    let (model, varmap) = LoraAarambhModel::from_tensors(
        &model_config,
        &base_tensors,
        &config.lora_config,
        false,
        &candle_device,
    )?;
    eprintln!(
        "grpo adapter params: {} / {} ({:.3}%)",
        model.adapter_param_count(),
        model.base_param_count(),
        model.trainable_ratio() * 100.0
    );

    let dataset = GrpoDataset::from_jsonl(&config.data_path)?;
    let metadata = AdapterMetadata::new(
        model_config,
        config.lora_config.clone(),
        Some(config.base_model_path.display().to_string()),
        false,
    );
    let save_metadata = GrpoSaveMetadata {
        grpo: config.grpo_config.clone(),
        train: config.train_config.clone(),
        verifier: config.verifier,
        reference_model: config.reference_model_path.display().to_string(),
    };
    let mut trainer = GrpoTrainer::new(
        model,
        varmap,
        reference,
        tokenizer,
        config.verifier.build(),
        dataset,
        config.grpo_config,
        config.train_config,
        config.output_dir,
        metadata,
        save_metadata,
        config.shuffle,
        candle_device,
    )?;
    trainer.train()
}

pub fn sample_group(
    model: &LoraAarambhModel,
    tokenizer: &BpeTokenizer,
    example: &GrpoExample,
    config: &GrpoConfig,
    seed: u64,
    device: &CandleDevice,
) -> Result<Vec<Rollout>> {
    config.validate()?;
    let template = ChatTemplate;
    let prompt = template.prefix(&example.prompt, None);
    let prompt_ids = tokenizer.encode(&prompt)?;
    if prompt_ids.is_empty() {
        return Err(AarambhError::Config(
            "GRPO prompt encoded to zero tokens".into(),
        ));
    }
    if prompt_ids.len() >= model.config().max_seq_len {
        return Err(AarambhError::Shape(format!(
            "GRPO prompt length {} leaves no room for completion in max_seq_len {}",
            prompt_ids.len(),
            model.config().max_seq_len
        )));
    }

    let mut rollouts = Vec::with_capacity(config.group_size);
    for group_idx in 0..config.group_size {
        let mut sampler = LocalSampler::new(
            config.temperature,
            config.top_k,
            config.top_p,
            seed ^ ((group_idx as u64 + 1) * 0x9E37_79B9_7F4A_7C15),
        );
        let mut thinking = LocalThinkingState::new(config.thinking, config.max_new_tokens);
        let mut token_ids = prompt_ids.clone();
        let mut completion = Vec::new();
        let mut finish_reason = RolloutFinish::MaxTokens;

        for _ in 0..config.max_new_tokens {
            if token_ids.len() >= model.config().max_seq_len {
                finish_reason = RolloutFinish::ContextLimit;
                break;
            }
            let next_id = if let Some(forced) = thinking.take_forced_token() {
                forced
            } else {
                let logits = next_token_logits(model, &token_ids, device)?;
                sampler.sample(&logits)?
            };
            token_ids.push(next_id);
            completion.push(next_id);
            thinking.on_token(next_id);
            if next_id == ENDOFTEXT_ID {
                finish_reason = RolloutFinish::Eos;
                break;
            }
        }

        if completion.is_empty() {
            finish_reason = RolloutFinish::MaxTokens;
            completion.push(ENDOFTEXT_ID);
        }
        let completion_text = tokenizer.decode(&completion)?;
        rollouts.push(Rollout {
            prompt_len: prompt_ids.len(),
            completion_token_ids: completion,
            completion_text,
            score: 0.0,
            advantage: 0.0,
            finish_reason,
        });
    }
    Ok(rollouts)
}

pub fn compute_advantages(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let mean = mean_f32(scores);
    let std = std_f32(scores, mean);
    if !std.is_finite() || std <= 1e-8 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|score| (*score - mean) / std).collect()
}

pub fn grpo_loss(
    policy_log_probs: &[Tensor],
    ref_log_probs: &[Tensor],
    advantages: &[f32],
    kl_coeff: f64,
) -> Result<Tensor> {
    if policy_log_probs.len() != ref_log_probs.len() || policy_log_probs.len() != advantages.len() {
        return Err(AarambhError::Shape(
            "policy_log_probs, ref_log_probs, and advantages must have equal length".into(),
        ));
    }
    let kl_terms = policy_log_probs
        .iter()
        .zip(ref_log_probs.iter())
        .map(|(policy, reference)| {
            let diff = (policy - &reference.detach())?;
            mean_tensor(&diff)
        })
        .collect::<Result<Vec<_>>>()?;
    grpo_loss_with_full_kl(policy_log_probs, &kl_terms, advantages, kl_coeff)
}

pub fn grpo_loss_with_full_kl(
    policy_log_probs: &[Tensor],
    kl_terms: &[Tensor],
    advantages: &[f32],
    kl_coeff: f64,
) -> Result<Tensor> {
    if policy_log_probs.len() != kl_terms.len() || policy_log_probs.len() != advantages.len() {
        return Err(AarambhError::Shape(
            "policy_log_probs, kl_terms, and advantages must have equal length".into(),
        ));
    }
    if policy_log_probs.is_empty() {
        return Err(AarambhError::Shape(
            "GRPO loss requires at least one rollout".into(),
        ));
    }
    let mut total = None;
    for ((policy, kl), advantage) in policy_log_probs
        .iter()
        .zip(kl_terms.iter())
        .zip(advantages.iter().copied())
    {
        let reward = mean_tensor(policy)?.affine(-(advantage as f64), 0.0)?;
        let kl = kl.affine(kl_coeff, 0.0)?;
        let term = (reward + kl)?;
        total = Some(match total {
            Some(acc) => (acc + term)?,
            None => term,
        });
    }
    Ok(total
        .expect("non-empty checked above")
        .affine(1.0 / policy_log_probs.len() as f64, 0.0)?)
}

fn replay_policy_terms(
    policy: &LoraAarambhModel,
    reference: &AarambhModel,
    tokenizer: &BpeTokenizer,
    example: &GrpoExample,
    rollout: &Rollout,
    device: &CandleDevice,
) -> Result<(Tensor, Tensor)> {
    let template = ChatTemplate;
    let prompt = template.prefix(&example.prompt, None);
    let mut full_ids = tokenizer.encode(&prompt)?;
    let prompt_len = full_ids.len();
    full_ids.extend_from_slice(&rollout.completion_token_ids);
    if full_ids.len() < 2 {
        return Err(AarambhError::Shape(
            "GRPO replay sequence must contain at least two tokens".into(),
        ));
    }
    if rollout.completion_token_ids.is_empty() {
        return Err(AarambhError::Shape(
            "GRPO rollout has no completion tokens".into(),
        ));
    }
    if full_ids.len() > policy.config().max_seq_len {
        return Err(AarambhError::Shape(format!(
            "GRPO replay sequence length {} exceeds max_seq_len {}",
            full_ids.len(),
            policy.config().max_seq_len
        )));
    }

    let input_ids = &full_ids[..full_ids.len() - 1];
    let input = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), device)?;
    let policy_logits = policy.forward_train(&input)?;
    let reference_logits = reference.forward_train(&input)?.detach();
    let selected = selected_completion_log_probs(
        &policy_logits,
        &rollout.completion_token_ids,
        prompt_len,
        device,
    )?;
    let kl = full_distribution_kl(
        &policy_logits,
        &reference_logits,
        prompt_len,
        rollout.completion_token_ids.len(),
    )?;
    Ok((selected, kl))
}

fn selected_completion_log_probs(
    logits: &Tensor,
    completion_token_ids: &[u32],
    prompt_len: usize,
    device: &CandleDevice,
) -> Result<Tensor> {
    let dims = logits.dims();
    if dims.len() != 3 {
        return Err(AarambhError::Shape(format!(
            "logits must have shape [batch, seq, vocab], got {dims:?}"
        )));
    }
    if prompt_len == 0 {
        return Err(AarambhError::Shape(
            "prompt_len must be greater than zero".into(),
        ));
    }
    let seq_len = dims[1];
    let vocab = dims[2];
    let start = prompt_len - 1;
    let len = completion_token_ids.len();
    if start + len > seq_len {
        return Err(AarambhError::Shape(format!(
            "completion positions [{start}, {}) exceed logits seq_len {seq_len}",
            start + len
        )));
    }
    let logits = logits.narrow(1, start, len)?.reshape((len, vocab))?;
    let labels = Tensor::from_vec(completion_token_ids.to_vec(), (len,), device)?;
    let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
    Ok(log_probs
        .gather(&labels.unsqueeze(1)?, 1)?
        .reshape((len,))?)
}

fn full_distribution_kl(
    policy_logits: &Tensor,
    reference_logits: &Tensor,
    prompt_len: usize,
    completion_len: usize,
) -> Result<Tensor> {
    if prompt_len == 0 {
        return Err(AarambhError::Shape(
            "prompt_len must be greater than zero".into(),
        ));
    }
    let dims = policy_logits.dims();
    if dims.len() != 3 || reference_logits.dims() != dims {
        return Err(AarambhError::Shape(format!(
            "policy/ref logits must have matching [batch, seq, vocab] shapes, got {:?} and {:?}",
            policy_logits.dims(),
            reference_logits.dims()
        )));
    }
    let seq_len = dims[1];
    let vocab = dims[2];
    let start = prompt_len - 1;
    if start + completion_len > seq_len {
        return Err(AarambhError::Shape(format!(
            "completion positions [{start}, {}) exceed logits seq_len {seq_len}",
            start + completion_len
        )));
    }
    let policy = policy_logits
        .narrow(1, start, completion_len)?
        .reshape((completion_len, vocab))?;
    let reference = reference_logits
        .narrow(1, start, completion_len)?
        .reshape((completion_len, vocab))?;
    let policy_log_probs = candle_nn::ops::log_softmax(&policy, 1)?;
    let reference_log_probs = candle_nn::ops::log_softmax(&reference, 1)?.detach();
    let policy_probs = policy_log_probs.exp()?;
    let diff = (&policy_log_probs - &reference_log_probs)?;
    let per_token_kl = (policy_probs * diff)?.sum(1)?;
    mean_tensor(&per_token_kl)
}

fn next_token_logits(
    model: &LoraAarambhModel,
    token_ids: &[u32],
    device: &CandleDevice,
) -> Result<Vec<f32>> {
    let input = Tensor::from_vec(token_ids.to_vec(), (1, token_ids.len()), device)?;
    let logits = model.forward_eval(&input)?;
    let vocab = logits.dims()[2];
    Ok(logits
        .narrow(1, token_ids.len() - 1, 1)?
        .reshape((vocab,))?
        .to_vec1::<f32>()?)
}

#[derive(Debug)]
struct LocalThinkingState {
    mode: GrpoThinkingMode,
    budget: usize,
    started: bool,
    closed: bool,
    in_thinking: bool,
    tokens_used: usize,
    pending_end: bool,
}

impl LocalThinkingState {
    fn new(mode: GrpoThinkingMode, max_new_tokens: usize) -> Self {
        let budget = if mode.is_enabled() {
            mode.budget().min(max_new_tokens.saturating_sub(32))
        } else {
            0
        };
        Self {
            mode,
            budget,
            started: false,
            closed: false,
            in_thinking: false,
            tokens_used: 0,
            pending_end: false,
        }
    }

    fn take_forced_token(&mut self) -> Option<u32> {
        if self.pending_end {
            self.pending_end = false;
            return Some(THINK_END_ID);
        }
        if self.mode.is_enabled() && !self.started && !self.closed {
            return Some(THINK_START_ID);
        }
        None
    }

    fn on_token(&mut self, token_id: u32) {
        if !self.mode.is_enabled() {
            return;
        }
        if token_id == THINK_START_ID && !self.started {
            self.started = true;
            self.in_thinking = true;
            if self.budget == 0 {
                self.pending_end = true;
            }
            return;
        }
        if !self.in_thinking {
            return;
        }
        if token_id == THINK_END_ID {
            self.in_thinking = false;
            self.closed = true;
            return;
        }
        self.tokens_used += 1;
        if self.tokens_used >= self.budget {
            self.pending_end = true;
        }
    }
}

#[derive(Debug)]
struct LocalSampler {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: StdRng,
}

impl LocalSampler {
    fn new(temperature: f32, top_k: Option<usize>, top_p: Option<f32>, seed: u64) -> Self {
        Self {
            temperature,
            top_k: top_k.filter(|k| *k > 0),
            top_p: top_p.filter(|p| *p > 0.0 && *p < 1.0),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(AarambhError::Shape("logits must be non-empty".into()));
        }
        if self.temperature <= f32::EPSILON {
            return Ok(argmax(logits) as u32);
        }
        let probs = filtered_probs(logits, self.temperature, self.top_k, self.top_p)?;
        let draw = self.rng.r#gen::<f32>();
        let mut cumulative = 0.0f32;
        for (idx, probability) in probs.iter().enumerate() {
            cumulative += *probability;
            if draw <= cumulative {
                return Ok(idx as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn filtered_probs(
    logits: &[f32],
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> Result<Vec<f32>> {
    let mut allowed = vec![true; logits.len()];
    if let Some(k) = top_k
        && k < logits.len()
    {
        let mut ranked = logits.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        for (idx, _) in ranked.into_iter().skip(k) {
            allowed[idx] = false;
        }
    }

    let masked_logits = logits
        .iter()
        .zip(allowed.iter())
        .map(|(logit, allowed)| {
            if *allowed {
                *logit / temperature
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect::<Vec<_>>();
    let mut probs = softmax(&masked_logits)?;

    if let Some(p) = top_p {
        let mut ranked = probs.iter().copied().enumerate().collect::<Vec<_>>();
        ranked.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        let mut keep = vec![false; probs.len()];
        let mut cumulative = 0.0f32;
        for (idx, probability) in ranked {
            keep[idx] = true;
            cumulative += probability;
            if cumulative >= p {
                break;
            }
        }
        for (idx, probability) in probs.iter_mut().enumerate() {
            if !keep[idx] {
                *probability = 0.0;
            }
        }
        renormalize(&mut probs)?;
    }
    Ok(probs)
}

fn softmax(logits: &[f32]) -> Result<Vec<f32>> {
    let max = logits
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .max_by(|a, b| a.total_cmp(b))
        .ok_or_else(|| AarambhError::Config("all logits are non-finite".into()))?;
    let mut probs = logits
        .iter()
        .map(|logit| {
            if logit.is_finite() {
                (*logit - max).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    renormalize(&mut probs)?;
    Ok(probs)
}

fn renormalize(probs: &mut [f32]) -> Result<()> {
    let sum = probs.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(AarambhError::Config(
            "sampling distribution has zero probability mass".into(),
        ));
    }
    for probability in probs {
        *probability /= sum;
    }
    Ok(())
}

fn mean_tensor(tensor: &Tensor) -> Result<Tensor> {
    if tensor.elem_count() == 0 {
        return Err(AarambhError::Shape(
            "cannot compute mean of empty tensor".into(),
        ));
    }
    Ok(tensor
        .sum_all()?
        .affine(1.0 / tensor.elem_count() as f64, 0.0)?)
}

fn mean_scalar_tensors(tensors: &[Tensor]) -> Result<f64> {
    if tensors.is_empty() {
        return Ok(0.0);
    }
    let mut total = 0.0;
    for tensor in tensors {
        total += tensor.to_scalar::<f32>()? as f64;
    }
    Ok(total / tensors.len() as f64)
}

fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

fn std_f32(values: &[f32], mean: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / values.len() as f32;
    variance.sqrt()
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value).map_err(AarambhError::Json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn advantages_are_zero_mean() {
        let advantages = compute_advantages(&[0.0, 1.0, 2.0, 3.0]);
        let mean = mean_f32(&advantages);
        assert!(mean.abs() < 1e-6, "mean was {mean}");
    }

    #[test]
    fn advantages_handle_zero_variance() {
        assert_eq!(compute_advantages(&[1.0, 1.0, 1.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn grpo_loss_is_finite() {
        let device = Device::Cpu;
        let policy_a = Tensor::new(&[-0.2f32, -0.4], &device).unwrap();
        let policy_b = Tensor::new(&[-0.7f32, -0.1], &device).unwrap();
        let ref_a = Tensor::new(&[-0.3f32, -0.5], &device).unwrap();
        let ref_b = Tensor::new(&[-0.6f32, -0.2], &device).unwrap();
        let loss = grpo_loss(&[policy_a, policy_b], &[ref_a, ref_b], &[1.0, -1.0], 0.01)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(loss.is_finite());
    }

    #[test]
    fn dataset_accepts_prompt_and_gsm8k_question_records() {
        let path =
            std::env::temp_dir().join(format!("aarambh_grpo_dataset_{}.jsonl", std::process::id()));
        fs::write(
            &path,
            "{\"prompt\":\"What is 2+2?\",\"ground_truth\":\"4\"}\n{\"question\":\"What is 3+5?\",\"answer\":\"work\\n#### 8\"}\n",
        )
        .unwrap();
        let dataset = GrpoDataset::from_jsonl(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.examples[1].ground_truth, "work\n#### 8");
    }

    #[test]
    fn local_sampler_top_k_masks_lower_ranked_tokens() {
        let mut sampler = LocalSampler::new(0.0, Some(1), Some(0.5), 42);
        assert_eq!(sampler.sample(&[0.1, 9.0, 0.2]).unwrap(), 1);
    }

    #[test]
    fn thinking_state_forces_start_and_end_when_budget_is_zero() {
        let mut state = LocalThinkingState::new(GrpoThinkingMode::Low, 4);
        assert_eq!(state.take_forced_token(), Some(THINK_START_ID));
        state.on_token(THINK_START_ID);
        assert_eq!(state.take_forced_token(), Some(THINK_END_ID));
        state.on_token(THINK_END_ID);
        assert_eq!(state.take_forced_token(), None);
    }

    #[test]
    fn selected_log_probs_have_completion_shape() {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(
            vec![
                5f32, 1., 0., 0., //
                0., 5., 0., 0., //
                0., 0., 5., 0., //
            ],
            (1, 3, 4),
            &device,
        )
        .unwrap();
        let picked = selected_completion_log_probs(&logits, &[1, 2], 2, &device).unwrap();
        assert_eq!(picked.dims(), &[2]);
    }

    #[test]
    fn full_distribution_kl_sums_vocab_then_averages_tokens() {
        let device = Device::Cpu;
        let policy = Tensor::from_vec(vec![0f32, 0.], (1, 1, 2), &device).unwrap();
        let reference =
            Tensor::from_vec(vec![0.9f32.ln(), 0.1f32.ln()], (1, 1, 2), &device).unwrap();
        let kl = full_distribution_kl(&policy, &reference, 1, 1)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(kl > 0.4 && kl < 0.7, "kl was {kl}");
    }
}
