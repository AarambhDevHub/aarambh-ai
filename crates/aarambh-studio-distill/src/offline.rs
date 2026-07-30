use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use aarambh_studio_core::{
    AarambhError, Configurable, DType, Device, ModelConfig, Result, TokenizerLike, TrainConfig,
};
use aarambh_studio_inference::InferenceEngine;
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::{BpeTokenizer, PAD_ID};
use aarambh_studio_train::loss::cross_entropy_loss;
use aarambh_studio_train::optim::clip_gradients;
use aarambh_studio_train::{
    AdamW, AdamWConfig, CosineScheduleWithWarmup, DsaTrainingConfig, GradMap, combine_mtp_losses,
    mtp_head_loss,
};
use candle_core::backprop::GradStore;
use candle_nn::{VarBuilder, VarMap};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::checkpoint::{DistillCheckpointManager, DistillState};
use crate::config::DistillConfig;
use crate::dataset::{OfflineDataset, OfflineExample, PromptDataset, ReplayBatch};
use crate::rollout::{StudentRollout, generate_student_rollouts};

/// Configuration for generating a static offline teacher-completion dataset.
#[derive(Debug, Clone)]
pub struct OfflinePrepareConfig {
    /// Frozen teacher architecture.
    pub teacher_model_config: ModelConfig,
    /// Frozen teacher checkpoint.
    pub teacher_model_path: PathBuf,
    /// Shared tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Prompt JSONL input.
    pub prompt_path: PathBuf,
    /// Offline completion JSONL output.
    pub output_path: PathBuf,
    /// Teacher generation device.
    pub device: Device,
    /// Teacher parameter dtype.
    pub dtype: DType,
    /// Generation settings; one rollout is produced per prompt.
    pub generation: DistillConfig,
    /// Deterministic generation seed.
    pub seed: u64,
}

/// Configuration for the matched full-weight offline distillation control.
#[derive(Debug, Clone)]
pub struct OfflineRunConfig {
    /// Student architecture.
    pub model_config: ModelConfig,
    /// Optimizer and checkpoint cadence.
    pub train_config: TrainConfig,
    /// DSA indexer teacher cadence and weight.
    pub dsa_training_config: DsaTrainingConfig,
    /// Trainable student SafeTensors checkpoint.
    pub student_model_path: PathBuf,
    /// Shared tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Static teacher-completion JSONL.
    pub data_path: PathBuf,
    /// Output checkpoint directory.
    pub output_dir: PathBuf,
    /// Student device.
    pub device: Device,
    /// Student dtype.
    pub dtype: DType,
    /// Shuffle static completions each epoch.
    pub shuffle: bool,
    /// Restore the latest output checkpoint.
    pub resume: bool,
}

/// Generate and save one frozen teacher completion per prompt.
pub fn prepare_offline_dataset(mut config: OfflinePrepareConfig) -> Result<()> {
    config.generation.rollouts_per_prompt = 1;
    config.generation.validate()?;
    let device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    config.teacher_model_config.vocab_size = tokenizer.vocab_size();
    let engine = InferenceEngine::from_paths_with_dtype(
        &config.teacher_model_path,
        &config.teacher_model_config,
        &config.tokenizer_path,
        device,
        config.dtype.to_candle(),
    )?;
    let prompts = PromptDataset::from_jsonl(&config.prompt_path)?;
    let rollouts = generate_student_rollouts(
        &engine,
        prompts.examples(),
        &config.generation,
        0,
        config.seed,
    )?;
    if let Some(parent) = config.output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(&config.output_path)?;
    let mut writer = BufWriter::new(file);
    for rollout in rollouts {
        if rollout.completion_text.trim().is_empty() {
            return Err(AarambhError::Config(format!(
                "teacher produced an empty offline completion for prompt '{}'",
                rollout.prompt_id
            )));
        }
        serde_json::to_writer(
            &mut writer,
            &OfflineExample {
                id: rollout.prompt_id,
                prompt: rollout.prompt,
                completion: rollout.completion_text,
            },
        )
        .map_err(AarambhError::Json)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

/// Train the matched completion-only offline distillation baseline.
pub fn run_offline_distill_from_config(mut config: OfflineRunConfig) -> Result<()> {
    config.dsa_training_config.validate()?;
    if config
        .student_model_path
        .extension()
        .and_then(|value| value.to_str())
        != Some("safetensors")
    {
        return Err(AarambhError::Config(
            "the trainable offline student must be a SafeTensors checkpoint".into(),
        ));
    }
    let device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    config.model_config.vocab_size = tokenizer.vocab_size();
    let mut varmap = VarMap::new();
    let model = AarambhModel::new(
        &config.model_config,
        VarBuilder::from_varmap(&varmap, config.dtype.to_candle(), &device),
    )?;
    varmap.load(&config.student_model_path)?;
    let engine = InferenceEngine::new(model, tokenizer.clone(), device.clone())?;
    let dataset = OfflineDataset::from_jsonl(&config.data_path)?;
    let rollouts = tokenize_offline(&engine, &tokenizer, &dataset)?;
    if config.train_config.batch_size == 0
        || config.train_config.grad_accum_steps == 0
        || config.train_config.max_steps == 0
        || config.train_config.max_epochs == 0
    {
        return Err(AarambhError::Config(
            "offline distillation training limits must be non-zero".into(),
        ));
    }
    let manifest = serde_json::json!({
        "schema_version": 1,
        "mode": "offline_teacher_nll",
        "student_model": config.student_model_path.display().to_string(),
        "student_model_config": config.model_config,
        "tokenizer": config.tokenizer_path.display().to_string(),
        "offline_data": config.data_path.display().to_string(),
        "train": config.train_config,
        "dsa_training": config.dsa_training_config,
        "shuffle": config.shuffle,
    });
    fs::create_dir_all(&config.output_dir)?;
    let checkpoint = DistillCheckpointManager::new(&config.output_dir);
    let mut optimizer = AdamW::from_varmap(&varmap, AdamWConfig::from(&config.train_config))?;
    let schedule = CosineScheduleWithWarmup::from_train_config(&config.train_config);
    let mut state = DistillState::default();
    reset_order(
        &mut state,
        rollouts.len(),
        config.shuffle,
        config.train_config.seed,
    );
    if config.resume {
        state = checkpoint
            .load_latest(&mut varmap, &mut optimizer, &manifest, &device)?
            .ok_or_else(|| {
                AarambhError::Checkpoint(
                    "offline distillation resume requested but no checkpoint exists".into(),
                )
            })?;
    }
    let mut pending = GradMap::new();
    while state.epoch < config.train_config.max_epochs && state.step < config.train_config.max_steps
    {
        if state.prompt_position >= state.prompt_order.len() {
            flush_offline(
                &mut pending,
                &mut optimizer,
                &schedule,
                &config.train_config,
                &mut state,
            )?;
            state.epoch += 1;
            if state.epoch < config.train_config.max_epochs {
                reset_order(
                    &mut state,
                    rollouts.len(),
                    config.shuffle,
                    config.train_config.seed,
                );
            }
            continue;
        }
        let end =
            (state.prompt_position + config.train_config.batch_size).min(state.prompt_order.len());
        let batch = state.prompt_order[state.prompt_position..end]
            .iter()
            .map(|index| rollouts[*index].clone())
            .collect::<Vec<_>>();
        state.prompt_position = end;
        let replay = ReplayBatch::from_rollouts(&batch, PAD_ID, &device)?;
        let collect_dsa_teacher = engine.model().config().dsa_config.is_some()
            && state
                .step
                .is_multiple_of(config.dsa_training_config.teacher_every_n_steps);
        let output = engine
            .model()
            .forward_train_with_aux_and_dsa_teacher(&replay.input_ids, collect_dsa_teacher)?;
        let nll = cross_entropy_loss(&output.logits, &replay.labels, &replay.completion_mask)?;
        let mut mtp_losses = Vec::with_capacity(engine.model().mtp_heads().len());
        for head_index in 0..engine.model().mtp_heads().len() {
            let prediction = engine.model().forward_mtp_head_train(
                head_index,
                &output.final_hidden_states,
                &replay.input_ids,
            )?;
            mtp_losses.push(mtp_head_loss(
                prediction,
                &replay.labels,
                &replay.completion_mask,
            )?);
        }
        let mtp_weight = engine
            .model()
            .config()
            .mtp
            .as_ref()
            .map(|mtp| mtp.aux_loss_weight)
            .unwrap_or(0.0);
        let mut loss = combine_mtp_losses(nll, mtp_losses, mtp_weight)?.total_loss;
        if let Some(moe) = &output.moe_aux_loss
            && let Some(config) = &engine.model().config().moe
            && config.aux_loss_weight > 0.0
        {
            loss = (&loss + &moe.affine(config.aux_loss_weight, 0.0)?)?;
        }
        if let Some(dsa) = &output.dsa_indexer_loss
            && config.dsa_training_config.indexer_loss_weight > 0.0
        {
            loss = (&loss + &dsa.affine(config.dsa_training_config.indexer_loss_weight, 0.0)?)?;
        }
        let loss_value = loss.to_scalar::<f32>()? as f64;
        if !loss_value.is_finite() {
            return Err(AarambhError::Config(
                "offline distillation produced a non-finite loss".into(),
            ));
        }
        let grads = loss
            .affine(1.0 / config.train_config.grad_accum_steps as f64, 0.0)?
            .backward()?;
        accumulate(&optimizer, &mut pending, &grads)?;
        state.micro_step += 1;
        state.train_loss = Some(loss_value);
        state.rollout_tokens += replay.completion_tokens();
        if state
            .micro_step
            .is_multiple_of(config.train_config.grad_accum_steps)
        {
            let lr = schedule.lr_at_step(state.step);
            let grad_norm = clip_gradients(&mut pending, config.train_config.clip_grad_norm)?;
            optimizer.step(&pending, lr)?;
            pending.clear();
            state.step += 1;
            if config.train_config.log_every_n_steps > 0
                && state
                    .step
                    .is_multiple_of(config.train_config.log_every_n_steps)
            {
                println!(
                    "offline_distill step={} loss={:.4} lr={:.6} grad_norm={:.4}",
                    state.step, loss_value, lr, grad_norm
                );
            }
            if config.train_config.save_every_n_steps > 0
                && state
                    .step
                    .is_multiple_of(config.train_config.save_every_n_steps)
            {
                checkpoint.save(&varmap, &optimizer, &state, &manifest)?;
            }
        }
    }
    flush_offline(
        &mut pending,
        &mut optimizer,
        &schedule,
        &config.train_config,
        &mut state,
    )?;
    checkpoint.save_final(&varmap, &optimizer, &state, &manifest)?;
    Ok(())
}

fn tokenize_offline(
    engine: &InferenceEngine,
    tokenizer: &BpeTokenizer,
    dataset: &OfflineDataset,
) -> Result<Vec<StudentRollout>> {
    dataset
        .examples()
        .iter()
        .map(|example| {
            let prompt_token_ids = engine.encode_prompt(&example.prompt)?;
            let mut completion_token_ids = tokenizer.encode(&example.completion)?;
            completion_token_ids.push(tokenizer.eos_token_id());
            if prompt_token_ids.len() + completion_token_ids.len()
                > engine.model().config().max_seq_len
            {
                return Err(AarambhError::Shape(format!(
                    "offline example '{}' exceeds student max_seq_len",
                    example.id
                )));
            }
            let loss_mask = vec![true; completion_token_ids.len()];
            Ok(StudentRollout {
                prompt_id: example.id.clone(),
                prompt: example.prompt.clone(),
                prompt_token_ids,
                completion_token_ids,
                completion_text: example.completion.clone(),
                loss_mask,
                rollout_index: 0,
                finish_reason: crate::rollout::RolloutFinish::Eos,
            })
        })
        .collect()
}

fn reset_order(state: &mut DistillState, count: usize, shuffle: bool, seed: u64) {
    state.prompt_position = 0;
    state.prompt_order = (0..count).collect();
    if shuffle {
        let mut rng = StdRng::seed_from_u64(seed ^ (state.epoch as u64).rotate_left(29));
        state.prompt_order.shuffle(&mut rng);
    }
}

fn accumulate(optimizer: &AdamW, pending: &mut GradMap, grads: &GradStore) -> Result<()> {
    let mut updates = Vec::new();
    for parameter in optimizer.parameters() {
        let Some(gradient) = grads.get(parameter.tensor()) else {
            continue;
        };
        let gradient = gradient.detach();
        let accumulated = match pending.get(parameter.name()) {
            Some(existing) => ((existing + &gradient)?).detach(),
            None => gradient,
        };
        updates.push((parameter.name().to_string(), accumulated));
    }
    if updates.is_empty() {
        return Err(AarambhError::Config(
            "offline distillation backward produced no gradients".into(),
        ));
    }
    pending.extend(updates);
    Ok(())
}

fn flush_offline(
    pending: &mut GradMap,
    optimizer: &mut AdamW,
    schedule: &CosineScheduleWithWarmup,
    config: &TrainConfig,
    state: &mut DistillState,
) -> Result<()> {
    if pending.is_empty() || state.step >= config.max_steps {
        return Ok(());
    }
    let lr = schedule.lr_at_step(state.step);
    let _ = clip_gradients(pending, config.clip_grad_norm)?;
    optimizer.step(pending, lr)?;
    pending.clear();
    state.step += 1;
    Ok(())
}
