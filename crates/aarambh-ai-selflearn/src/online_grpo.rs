use std::fs;
use std::path::{Path, PathBuf};

use aarambh_ai_core::{AarambhError, ModelConfig, Result, TokenizerLike, TrainConfig};
use aarambh_ai_finetune::sft::build_loss_mask;
use aarambh_ai_finetune::{
    AdapterMetadata, ChatTemplate, GrpoConfig, GrpoExample, GrpoThinkingMode, LoraAarambhModel,
    LoraConfig, Rollout, SftExample, compute_advantages, sample_group,
};
use aarambh_ai_inference::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep,
    ThinkingController, ThinkingMode,
};
use aarambh_ai_model::AarambhModel;
use aarambh_ai_tokenizer::{
    BOS_ID, BpeTokenizer, ENDOFTEXT_ID, PAD_ID, THINK_END_ID, THINK_START_ID,
};
use aarambh_ai_train::optim::clip_gradients;
use aarambh_ai_train::{AdamW, AdamWConfig, GradMap, cross_entropy_loss};
use aarambh_ai_weights::load_any_model_with_dtype;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device as CandleDevice, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

use crate::config::{OnlineGrpoConfig, SelfLearnMode};
use crate::critique::CritiqueGenerator;

#[derive(Debug, Clone)]
pub struct OnlineGrpoBuildConfig {
    pub model_config: ModelConfig,
    pub base_model_path: PathBuf,
    pub reference_model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub state_dir: PathBuf,
    pub config: OnlineGrpoConfig,
    pub mode: SelfLearnMode,
    pub device: CandleDevice,
    pub dtype: DType,
    pub seed: u64,
}

#[derive(Debug)]
pub struct OnlineUpdate {
    pub output: GenerationOutput,
    pub verifier_score: Option<f32>,
    pub pending_grads: GradMap,
    pub used_grpo: bool,
}

pub struct OnlineGrpo {
    model: LoraAarambhModel,
    varmap: VarMap,
    reference: AarambhModel,
    tokenizer: BpeTokenizer,
    optimizer: AdamW,
    train_config: TrainConfig,
    config: OnlineGrpoConfig,
    mode: SelfLearnMode,
    pending_grads: GradMap,
    pending_grad_steps: usize,
    step_count: usize,
    state_dir: PathBuf,
    metadata: AdapterMetadata,
    device: CandleDevice,
    seed: u64,
}

impl OnlineGrpo {
    pub fn from_paths(mut build: OnlineGrpoBuildConfig) -> Result<Self> {
        let tokenizer = BpeTokenizer::from_pretrained(&build.tokenizer_path)?;
        tokenizer.validate_special_tokens()?;
        build.model_config.vocab_size = tokenizer.vocab_size();

        let base = load_any_model_with_dtype(
            &build.base_model_path,
            &build.model_config,
            &build.device,
            build.dtype,
        )?;
        let base_tensors = base.named_tensors();
        drop(base);
        let reference = load_any_model_with_dtype(
            &build.reference_model_path,
            &build.model_config,
            &build.device,
            build.dtype,
        )?;

        let lora_config = LoraConfig {
            rank: build.config.lora_rank,
            alpha: build.config.lora_rank as f64 * 2.0,
            dropout: 0.05,
            ..Default::default()
        };
        let (model, mut varmap) = LoraAarambhModel::from_tensors(
            &build.model_config,
            &base_tensors,
            &lora_config,
            false,
            &build.device,
        )?;
        let metadata = AdapterMetadata::new(
            build.model_config.clone(),
            lora_config,
            Some(build.base_model_path.display().to_string()),
            false,
        );
        if adapter_path(&build.state_dir).exists() {
            varmap.load(adapter_path(&build.state_dir))?;
        }

        let mut train_config = TrainConfig {
            lr: build.config.online_lr,
            batch_size: 1,
            grad_accum_steps: 1,
            max_epochs: 1,
            max_steps: usize::MAX,
            warmup_steps: 0,
            min_lr_ratio: 1.0,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 0,
            log_every_n_steps: 0,
            eval_steps: 0,
            seed: build.seed,
            checkpoint_dir: build.state_dir.clone(),
        };
        if build.mode == SelfLearnMode::Cpu {
            train_config.grad_accum_steps = usize::MAX;
        }
        let optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&train_config))?;
        if optimizer.parameters().is_empty() {
            return Err(AarambhError::Config(
                "self-learn LoRA target_modules produced zero trainable tensors".into(),
            ));
        }

        let mut this = Self {
            model,
            varmap,
            reference,
            tokenizer,
            optimizer,
            train_config,
            config: build.config,
            mode: build.mode,
            pending_grads: GradMap::new(),
            pending_grad_steps: 0,
            step_count: 0,
            state_dir: build.state_dir,
            metadata,
            device: build.device,
            seed: build.seed,
        };
        this.load_state()?;
        Ok(this)
    }

    pub fn tokenizer(&self) -> &BpeTokenizer {
        &self.tokenizer
    }

    pub fn step_count(&self) -> usize {
        self.step_count
    }

    pub fn pending_grads_count(&self) -> usize {
        self.pending_grads.len()
    }

    pub fn pending_grad_steps(&self) -> usize {
        self.pending_grad_steps
    }

    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    pub fn generate_update(
        &mut self,
        prompt: &str,
        generate_cfg: GenerationConfig,
        verifier: Option<&dyn aarambh_ai_finetune::Verifier>,
        ground_truth: Option<&str>,
    ) -> Result<OnlineUpdate> {
        if self.mode.is_enabled()
            && let (Some(verifier), Some(ground_truth)) = (verifier, ground_truth)
        {
            return self.generate_grpo_update(prompt, generate_cfg, verifier, ground_truth);
        }
        let output = generate_lora(
            &self.model,
            &self.tokenizer,
            prompt,
            generate_cfg,
            &self.device,
            |_| Ok(()),
        )?;
        Ok(OnlineUpdate {
            output,
            verifier_score: None,
            pending_grads: GradMap::new(),
            used_grpo: false,
        })
    }

    pub fn generate_text_with_lora(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<String> {
        Ok(generate_lora(
            &self.model,
            &self.tokenizer,
            prompt,
            config,
            &self.device,
            |_| Ok(()),
        )?
        .text)
    }

    pub fn commit_update(&mut self, update: OnlineUpdate) -> Result<Option<f64>> {
        if update.pending_grads.is_empty() {
            self.step_count += 1;
            self.save_state()?;
            return Ok(None);
        }
        match self.mode {
            SelfLearnMode::Cpu if self.config.skip_inline_on_cpu => {
                merge_grad_maps(&mut self.pending_grads, update.pending_grads)?;
                self.pending_grad_steps += 1;
                self.step_count += 1;
                self.save_state()?;
                Ok(None)
            }
            SelfLearnMode::Gpu | SelfLearnMode::Cpu => {
                let mut grads = update.pending_grads;
                let grad_norm = clip_gradients(&mut grads, self.train_config.clip_grad_norm)?;
                self.optimizer.step(&grads, self.config.online_lr)?;
                self.step_count += 1;
                self.save_state()?;
                Ok(Some(grad_norm))
            }
            SelfLearnMode::Disabled => Ok(None),
        }
    }

    pub fn flush_pending_gradients(&mut self) -> Result<Option<f64>> {
        if self.pending_grads.is_empty() {
            return Ok(None);
        }
        let divisor = self.pending_grad_steps.max(1) as f64;
        for grad in self.pending_grads.values_mut() {
            *grad = grad.affine(1.0 / divisor, 0.0)?.detach();
        }
        let grad_norm = clip_gradients(&mut self.pending_grads, self.train_config.clip_grad_norm)?;
        self.optimizer
            .step(&self.pending_grads, self.config.online_lr)?;
        self.pending_grads.clear();
        self.pending_grad_steps = 0;
        self.save_state()?;
        Ok(Some(grad_norm))
    }

    pub fn replay_sft_batch(
        &mut self,
        examples: &[SftExample],
        batch_size: usize,
    ) -> Result<Option<f64>> {
        if examples.is_empty() {
            return Ok(None);
        }
        if !self.pending_grads.is_empty() {
            let _ = self.flush_pending_gradients()?;
        }

        let max_seq_len = self.model.config().max_seq_len;
        let encoded = examples
            .iter()
            .map(|example| encode_replay_sft(example, &self.tokenizer, max_seq_len))
            .collect::<Result<Vec<_>>>()?;
        if encoded.is_empty() {
            return Ok(None);
        }

        let effective_batch = batch_size.max(1).min(encoded.len());
        let mut norm_sum = 0.0;
        let mut steps = 0usize;
        for chunk in encoded.chunks(effective_batch) {
            let (input_ids, labels, loss_mask) =
                replay_batch_to_tensors(chunk, max_seq_len, &self.device)?;
            let logits = self.model.forward_train(&input_ids)?;
            let loss = cross_entropy_loss(&logits, &labels, &loss_mask)?;
            let raw_grads = loss.backward()?;
            let mut grads = collect_grads(&self.optimizer, &raw_grads)?;
            if grads.is_empty() {
                return Err(AarambhError::Config(
                    "replay SFT backward produced no LoRA parameter gradients".into(),
                ));
            }
            let grad_norm = clip_gradients(&mut grads, self.train_config.clip_grad_norm)?;
            self.optimizer.step(&grads, self.config.online_lr)?;
            norm_sum += grad_norm;
            steps += 1;
            self.step_count += 1;
        }
        self.save_state()?;
        Ok(Some(norm_sum / steps as f64))
    }

    pub fn save_state(&self) -> Result<()> {
        fs::create_dir_all(&self.state_dir)?;
        aarambh_ai_finetune::save_adapter(&self.varmap, &self.metadata, &self.state_dir)?;
        self.optimizer
            .save_state(self.state_dir.join("optimizer.safetensors"))?;
        if self.pending_grads.is_empty() {
            let _ = fs::remove_file(pending_grads_path(&self.state_dir));
        } else {
            candle_core::safetensors::save(
                &self.pending_grads,
                pending_grads_path(&self.state_dir),
            )?;
        }
        write_json(
            self.state_dir.join("selflearn_state.json"),
            &OnlineState {
                step_count: self.step_count,
                mode: self.mode,
                pending_grads: self.pending_grads.len(),
                pending_grad_steps: Some(self.pending_grad_steps),
            },
        )
    }

    fn load_state(&mut self) -> Result<()> {
        let optimizer_path = self.state_dir.join("optimizer.safetensors");
        if optimizer_path.exists() {
            self.optimizer.load_state(&optimizer_path, &self.device)?;
        }
        let pending_path = pending_grads_path(&self.state_dir);
        if pending_path.exists() {
            self.pending_grads = candle_core::safetensors::load(&pending_path, &self.device)?;
        }
        let state_path = self.state_dir.join("selflearn_state.json");
        if state_path.exists() {
            let file = fs::File::open(state_path)?;
            let state: OnlineState = serde_json::from_reader(file)?;
            self.step_count = state.step_count;
            self.pending_grad_steps = state
                .pending_grad_steps
                .unwrap_or_else(|| usize::from(!self.pending_grads.is_empty()));
        }
        Ok(())
    }

    fn generate_grpo_update(
        &mut self,
        prompt: &str,
        generate_cfg: GenerationConfig,
        verifier: &dyn aarambh_ai_finetune::Verifier,
        ground_truth: &str,
    ) -> Result<OnlineUpdate> {
        let grpo_config = GrpoConfig {
            group_size: self.config.n_completions,
            kl_coeff: self.config.kl_coeff,
            max_new_tokens: generate_cfg
                .max_new_tokens
                .min(self.config.max_new_tokens)
                .max(1),
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            top_k: self.config.top_k,
            thinking: thinking_to_grpo(generate_cfg.thinking_mode),
        };
        let example = GrpoExample {
            prompt: prompt.to_string(),
            ground_truth: ground_truth.to_string(),
        };
        let mut rollouts = sample_group(
            &self.model,
            &self.tokenizer,
            &example,
            &grpo_config,
            self.seed ^ self.step_count as u64,
            &self.device,
        )?;
        for rollout in &mut rollouts {
            rollout.score = verifier.score(&rollout.completion_text, ground_truth);
        }
        let scores = rollouts
            .iter()
            .map(|rollout| rollout.score)
            .collect::<Vec<_>>();
        let advantages = compute_advantages(&scores);
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let mut policy_terms = Vec::new();
        let mut kl_terms = Vec::new();
        for rollout in &rollouts {
            let (policy, kl) = replay_policy_terms(
                &self.model,
                &self.reference,
                &self.tokenizer,
                &example,
                rollout,
                &self.device,
            )?;
            policy_terms.push(policy);
            kl_terms.push(kl);
        }
        let grads = if advantages.iter().any(|advantage| advantage.abs() > 1e-8) {
            let loss = aarambh_ai_finetune::grpo::grpo_loss_with_full_kl(
                &policy_terms,
                &kl_terms,
                &advantages,
                self.config.kl_coeff,
            )?;
            let raw_grads = loss.backward()?;
            collect_grads(&self.optimizer, &raw_grads)?
        } else {
            GradMap::new()
        };
        let output = rollout_to_generation_output(&rollouts[best_idx], &self.tokenizer)?;
        Ok(OnlineUpdate {
            output,
            verifier_score: Some(scores[best_idx]),
            pending_grads: grads,
            used_grpo: true,
        })
    }
}

impl CritiqueGenerator for OnlineGrpo {
    fn generate_text(&mut self, prompt: &str, config: GenerationConfig) -> Result<String> {
        self.generate_text_with_lora(prompt, config)
    }
}

pub fn generate_lora<F>(
    model: &LoraAarambhModel,
    tokenizer: &BpeTokenizer,
    prompt: &str,
    mut config: GenerationConfig,
    device: &CandleDevice,
    mut on_step: F,
) -> Result<GenerationOutput>
where
    F: FnMut(&GenerationStep) -> Result<()>,
{
    let mut prompt_ids = tokenizer.encode(prompt)?;
    if prompt_ids.is_empty() {
        prompt_ids.push(BOS_ID);
    }
    if prompt_ids.len() >= model.config().max_seq_len {
        return Err(AarambhError::Shape(format!(
            "prompt has {} tokens but model max_seq_len is {}",
            prompt_ids.len(),
            model.config().max_seq_len
        )));
    }
    let max_new_tokens = config
        .max_new_tokens
        .min(model.config().max_seq_len - prompt_ids.len());
    let mut token_ids = prompt_ids;
    let mut generated_ids = Vec::with_capacity(max_new_tokens);
    let mut raw_text = String::new();
    let mut answer_text = String::new();
    let mut thinking_text = String::new();
    let mut answer_token_ids = Vec::new();
    let mut thinking_token_ids = Vec::new();
    let mut steps = Vec::with_capacity(max_new_tokens);
    let mut finish_reason = FinishReason::MaxTokens;
    let mut thinking = ThinkingController::for_generation(config.thinking_mode, max_new_tokens);

    for step in 0..max_new_tokens {
        let forced = thinking.take_forced_token();
        let mut was_forced = forced.is_some();
        let token_id = match forced {
            Some(force) => force.token_id(),
            None => {
                let logits = next_token_logits(model, &token_ids, device)?;
                let sampled = config.sampler.sample(&logits)?;
                if sampled == ENDOFTEXT_ID && thinking.in_thinking_block() {
                    was_forced = true;
                    THINK_END_ID
                } else {
                    sampled
                }
            }
        };
        if token_id == ENDOFTEXT_ID && !thinking.in_thinking_block() {
            finish_reason = FinishReason::EosToken;
            break;
        }
        let phase = phase_for_token(&thinking, config.thinking_mode, token_id);
        let token_text = tokenizer.decode(&[token_id])?;
        let generation_step = GenerationStep {
            step: step + 1,
            token_id,
            token_text: token_text.clone(),
            candidates: Vec::new(),
            phase,
            forced: was_forced,
        };
        on_step(&generation_step)?;
        let _ = thinking.on_token(token_id);
        token_ids.push(token_id);
        generated_ids.push(token_id);
        raw_text.push_str(&token_text);
        if phase == GenerationPhase::Thinking && !is_thinking_marker(token_id) {
            thinking_text.push_str(&token_text);
            thinking_token_ids.push(token_id);
        } else if phase == GenerationPhase::Answer {
            answer_text.push_str(&token_text);
            answer_token_ids.push(token_id);
        }
        steps.push(generation_step);
    }

    if generated_ids.len() == max_new_tokens && token_ids.len() == model.config().max_seq_len {
        finish_reason = FinishReason::ContextLimit;
    }
    Ok(GenerationOutput {
        text: answer_text.clone(),
        raw_text,
        thinking_text,
        answer_text,
        token_ids: generated_ids,
        thinking_token_ids,
        answer_token_ids,
        thinking_tokens: thinking.tokens_used(),
        finish_reason,
        steps,
    })
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
            "online GRPO replay sequence must contain at least two tokens".into(),
        ));
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
    let seq_len = dims[1];
    let vocab = dims[2];
    let start = prompt_len.saturating_sub(1);
    let len = completion_token_ids.len();
    if start + len > seq_len {
        return Err(AarambhError::Shape(
            "completion positions exceed logits sequence length".into(),
        ));
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
    let dims = policy_logits.dims();
    if dims.len() != 3 || reference_logits.dims() != dims {
        return Err(AarambhError::Shape(
            "policy/ref logits must have matching [batch, seq, vocab] shapes".into(),
        ));
    }
    let vocab = dims[2];
    let start = prompt_len.saturating_sub(1);
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

fn collect_grads(optimizer: &AdamW, grads: &GradStore) -> Result<GradMap> {
    let mut updates = GradMap::new();
    for param in optimizer.parameters() {
        if let Some(grad) = grads.get(param.tensor()) {
            updates.insert(param.name().to_string(), grad.detach());
        }
    }
    Ok(updates)
}

struct ReplaySftSequence {
    input_ids: Vec<u32>,
    labels: Vec<u32>,
    loss_mask: Vec<u32>,
}

fn encode_replay_sft(
    example: &SftExample,
    tokenizer: &BpeTokenizer,
    max_seq_len: usize,
) -> Result<ReplaySftSequence> {
    if max_seq_len == 0 {
        return Err(AarambhError::Config("max_seq_len must be non-zero".into()));
    }
    let template = ChatTemplate;
    let prefix = template.prefix(&example.instruction, example.input.as_deref());
    let target = template.target(&example.response);
    let prefix_ids = tokenizer.encode(&prefix)?;
    let target_ids = tokenizer.encode(&target)?;
    if target_ids.is_empty() {
        return Err(AarambhError::Config(
            "replay SFT target encoded to zero tokens".into(),
        ));
    }
    let prefix_len = prefix_ids.len();
    let mut ids = prefix_ids;
    ids.extend(target_ids);
    if ids.len() > max_seq_len + 1 {
        ids.truncate(max_seq_len + 1);
    }
    if ids.len() < 2 {
        return Err(AarambhError::Config(
            "replay SFT sequence must contain at least two tokens".into(),
        ));
    }
    let loss_mask = build_loss_mask(prefix_len, ids.len());
    if loss_mask.iter().all(|value| *value == 0) {
        return Err(AarambhError::Config(
            "replay SFT sequence has no assistant tokens after truncation".into(),
        ));
    }
    Ok(ReplaySftSequence {
        input_ids: ids[..ids.len() - 1].to_vec(),
        labels: ids[1..].to_vec(),
        loss_mask,
    })
}

fn replay_batch_to_tensors(
    batch: &[ReplaySftSequence],
    max_seq_len: usize,
    device: &CandleDevice,
) -> Result<(Tensor, Tensor, Tensor)> {
    let mut input_ids = Vec::with_capacity(batch.len() * max_seq_len);
    let mut labels = Vec::with_capacity(batch.len() * max_seq_len);
    let mut loss_mask = Vec::with_capacity(batch.len() * max_seq_len);
    for sequence in batch {
        push_padded(&mut input_ids, &sequence.input_ids, max_seq_len, PAD_ID);
        push_padded(&mut labels, &sequence.labels, max_seq_len, PAD_ID);
        push_padded(&mut loss_mask, &sequence.loss_mask, max_seq_len, 0);
    }
    Ok((
        Tensor::from_vec(input_ids, (batch.len(), max_seq_len), device)?,
        Tensor::from_vec(labels, (batch.len(), max_seq_len), device)?,
        Tensor::from_vec(loss_mask, (batch.len(), max_seq_len), device)?,
    ))
}

fn push_padded(dst: &mut Vec<u32>, values: &[u32], max_len: usize, pad: u32) {
    let take = values.len().min(max_len);
    dst.extend_from_slice(&values[..take]);
    dst.extend(std::iter::repeat_n(pad, max_len - take));
}

fn merge_grad_maps(target: &mut GradMap, source: GradMap) -> Result<()> {
    for (name, grad) in source {
        let next = match target.get(&name) {
            Some(existing) => (existing + &grad)?.detach(),
            None => grad,
        };
        target.insert(name, next);
    }
    Ok(())
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

fn phase_for_token(
    thinking: &ThinkingController,
    thinking_mode: ThinkingMode,
    token_id: u32,
) -> GenerationPhase {
    if !thinking_mode.is_enabled() {
        return GenerationPhase::Answer;
    }
    if thinking.in_thinking_block() || (!thinking.has_started() && token_id == THINK_START_ID) {
        GenerationPhase::Thinking
    } else {
        GenerationPhase::Answer
    }
}

fn is_thinking_marker(token_id: u32) -> bool {
    token_id == THINK_START_ID || token_id == THINK_END_ID
}

fn thinking_to_grpo(mode: ThinkingMode) -> GrpoThinkingMode {
    match mode {
        ThinkingMode::None => GrpoThinkingMode::None,
        ThinkingMode::Low => GrpoThinkingMode::Low,
        ThinkingMode::Medium => GrpoThinkingMode::Medium,
        ThinkingMode::High => GrpoThinkingMode::High,
    }
}

fn rollout_to_generation_output(
    rollout: &Rollout,
    tokenizer: &BpeTokenizer,
) -> Result<GenerationOutput> {
    let steps = rollout
        .completion_token_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, token_id)| {
            Ok(GenerationStep {
                step: idx + 1,
                token_id,
                token_text: tokenizer.decode(&[token_id])?,
                candidates: Vec::new(),
                phase: GenerationPhase::Answer,
                forced: false,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GenerationOutput {
        text: rollout.completion_text.clone(),
        raw_text: rollout.completion_text.clone(),
        thinking_text: String::new(),
        answer_text: rollout.completion_text.clone(),
        token_ids: rollout.completion_token_ids.clone(),
        thinking_token_ids: Vec::new(),
        answer_token_ids: rollout.completion_token_ids.clone(),
        thinking_tokens: 0,
        finish_reason: match rollout.finish_reason {
            aarambh_ai_finetune::RolloutFinish::Eos => FinishReason::EosToken,
            aarambh_ai_finetune::RolloutFinish::MaxTokens => FinishReason::MaxTokens,
            aarambh_ai_finetune::RolloutFinish::ContextLimit => FinishReason::ContextLimit,
        },
        steps,
    })
}

fn adapter_path(state_dir: &Path) -> PathBuf {
    state_dir.join("adapter.safetensors")
}

fn pending_grads_path(state_dir: &Path) -> PathBuf {
    state_dir.join("pending_grads.safetensors")
}

#[derive(Debug, Serialize, Deserialize)]
struct OnlineState {
    step_count: usize,
    mode: SelfLearnMode,
    pending_grads: usize,
    #[serde(default)]
    pending_grad_steps: Option<usize>,
}

fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let file = fs::File::create(path.as_ref())?;
    serde_json::to_writer_pretty(file, value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use aarambh_ai_tokenizer::Vocab;
    use candle_core::Device;
    use std::collections::HashMap;

    #[test]
    fn merge_grad_maps_adds_matching_entries() {
        let device = Device::Cpu;
        let a = Tensor::new(&[1f32, 2.], &device).unwrap();
        let b = Tensor::new(&[3f32, 4.], &device).unwrap();
        let mut target = GradMap::new();
        target.insert("x".into(), a);
        let mut source = GradMap::new();
        source.insert("x".into(), b);
        merge_grad_maps(&mut target, source).unwrap();
        assert_eq!(target["x"].to_vec1::<f32>().unwrap(), vec![4.0, 6.0]);
    }

    #[test]
    fn empty_prompt_generation_uses_bos_shape() {
        let tokenizer = BpeTokenizer {
            vocab: Vocab {
                token_to_id: HashMap::from([("o".into(), 1u32), ("k".into(), 2u32)]),
                id_to_token: vec!["".into(), "o".into(), "k".into()],
            },
            merges: Vec::new(),
            merge_rank: HashMap::new(),
        };
        let output = rollout_to_generation_output(
            &Rollout {
                prompt_len: 1,
                completion_token_ids: vec![1, 2],
                completion_text: "ok".into(),
                score: 1.0,
                advantage: 0.0,
                finish_reason: aarambh_ai_finetune::RolloutFinish::MaxTokens,
            },
            &tokenizer,
        )
        .unwrap();
        assert_eq!(output.text, "ok");
        assert_eq!(output.token_ids, vec![1, 2]);
        assert_eq!(output.steps[0].token_text, "o");
    }
}
