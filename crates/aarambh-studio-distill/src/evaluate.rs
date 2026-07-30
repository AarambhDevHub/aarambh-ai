use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use aarambh_studio_core::{AarambhError, DType, Device, ModelConfig, Result, TokenizerLike};
use aarambh_studio_inference::InferenceEngine;
use aarambh_studio_model::AarambhModel;
use aarambh_studio_tokenizer::{BpeTokenizer, PAD_ID};
use serde::{Deserialize, Serialize};

use crate::config::{DistillConfig, DistillObjective};
use crate::dataset::{PromptDataset, ReplayBatch, ScoredReferenceDataset};
use crate::distill_loss::soft_kl_loss;
use crate::rollout::generate_student_rollouts;
use crate::teacher_score::{
    LocalCheckpointTeacher, ScoredDatasetTeacher, TeacherScorer, TeacherSignal,
};
use crate::trainer::TeacherSourceConfig;

/// Configuration for held-out fresh-rollout distillation evaluation.
#[derive(Debug, Clone)]
pub struct DistillEvalConfig {
    /// Student architecture.
    pub model_config: ModelConfig,
    /// Student checkpoint to evaluate.
    pub student_model_path: PathBuf,
    /// Shared tokenizer JSON.
    pub tokenizer_path: PathBuf,
    /// Held-out prompt JSONL.
    pub prompt_path: PathBuf,
    /// Teacher backend used for scoring.
    pub teacher: TeacherSourceConfig,
    /// Student rollout and objective settings.
    pub distill_config: DistillConfig,
    /// Student evaluation device.
    pub device: Device,
    /// Student evaluation dtype.
    pub dtype: DType,
    /// Optional limit on held-out prompts.
    pub max_prompts: Option<usize>,
    /// Optional JSON report path.
    pub output_json: Option<PathBuf>,
    /// Optional Markdown report path.
    pub output_markdown: Option<PathBuf>,
    /// Deterministic rollout seed.
    pub seed: u64,
}

/// Fresh-rollout teacher-alignment report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillEvalReport {
    /// Teacher backend identifier.
    pub teacher_backend: String,
    /// Distillation objective used for scoring.
    pub objective: DistillObjective,
    /// Number of held-out prompts.
    pub prompts: usize,
    /// Number of generated rollouts.
    pub rollouts: usize,
    /// Number of trainable completion tokens.
    pub completion_tokens: usize,
    /// Mean scalar teacher reward.
    pub reward_mean: f64,
    /// Standard deviation of scalar teacher rewards.
    pub reward_std: f64,
    /// Teacher-to-student forward KL for local soft targets.
    pub teacher_student_kl: Option<f64>,
    /// Mean completion length in tokens.
    pub mean_completion_tokens: f64,
    /// End-to-end rollout and scoring throughput.
    pub tokens_per_second: f64,
}

/// Evaluate a checkpoint on fresh student rollouts scored by the configured teacher.
pub fn evaluate_distillation(mut config: DistillEvalConfig) -> Result<DistillEvalReport> {
    config.distill_config.validate()?;
    if matches!(config.teacher, TeacherSourceConfig::Dataset { .. })
        && config.distill_config.objective != DistillObjective::Reward
    {
        return Err(AarambhError::Config(
            "scored-reference evaluation requires objective=reward".into(),
        ));
    }
    let device = config.device.to_candle()?;
    let tokenizer = BpeTokenizer::from_pretrained(&config.tokenizer_path)?;
    tokenizer.validate_special_tokens()?;
    config.model_config.vocab_size = tokenizer.vocab_size();
    let engine = InferenceEngine::from_paths_with_dtype(
        &config.student_model_path,
        &config.model_config,
        &config.tokenizer_path,
        device,
        config.dtype.to_candle(),
    )?;
    let dataset = PromptDataset::from_jsonl(&config.prompt_path)?;
    let prompt_limit = config
        .max_prompts
        .unwrap_or(dataset.len())
        .min(dataset.len());
    if prompt_limit == 0 {
        return Err(AarambhError::Config(
            "distillation evaluation prompt limit must be non-zero".into(),
        ));
    }
    let prompts = &dataset.examples()[..prompt_limit];
    let scorer: Box<dyn TeacherScorer> = match &mut config.teacher {
        TeacherSourceConfig::Local {
            model_path,
            model_config,
            device,
            dtype,
        } => {
            model_config.vocab_size = tokenizer.vocab_size();
            AarambhModel::validate_config(model_config)?;
            Box::new(LocalCheckpointTeacher::from_paths(
                model_path,
                model_config,
                device.to_candle()?,
                dtype.to_candle(),
            )?)
        }
        TeacherSourceConfig::Dataset { data_path } => {
            let references = ScoredReferenceDataset::from_jsonl(data_path)?;
            references.validate_prompt_coverage(&dataset)?;
            Box::new(ScoredDatasetTeacher::new(references, tokenizer.clone())?)
        }
    };
    let started = Instant::now();
    let rollouts =
        generate_student_rollouts(&engine, prompts, &config.distill_config, 0, config.seed)?;
    let replay = ReplayBatch::from_rollouts(&rollouts, PAD_ID, engine.device())?;
    let signal = match config.distill_config.objective {
        DistillObjective::SoftKl => TeacherSignal::SoftLogits,
        DistillObjective::Reward => TeacherSignal::Reward,
    };
    let feedback = scorer.score_batch(&rollouts, &replay, signal)?;
    let rewards = feedback
        .scores
        .iter()
        .map(|score| score.reward)
        .collect::<Vec<_>>();
    let reward_mean = mean(&rewards);
    let reward_std = stddev(&rewards, reward_mean as f32);
    let teacher_student_kl = match config.distill_config.objective {
        DistillObjective::SoftKl => {
            let teacher_logits = feedback.packed_logits.as_ref().ok_or_else(|| {
                AarambhError::Config("soft-KL evaluation received no teacher logits".into())
            })?;
            let student_logits = engine.model().forward_train(&replay.input_ids)?;
            let packed = replay.pack_logits(&student_logits)?;
            Some(
                soft_kl_loss(
                    &packed,
                    teacher_logits,
                    config.distill_config.teacher_temperature,
                )?
                .loss
                .to_scalar::<f32>()? as f64,
            )
        }
        DistillObjective::Reward => None,
    };
    let elapsed = started.elapsed().as_secs_f64().max(f64::EPSILON);
    let report = DistillEvalReport {
        teacher_backend: scorer.backend_name().to_string(),
        objective: config.distill_config.objective,
        prompts: prompt_limit,
        rollouts: rollouts.len(),
        completion_tokens: replay.completion_tokens(),
        reward_mean,
        reward_std,
        teacher_student_kl,
        mean_completion_tokens: replay.completion_tokens() as f64 / rollouts.len() as f64,
        tokens_per_second: replay.completion_tokens() as f64 / elapsed,
    };
    if let Some(path) = &config.output_json {
        write_json(path, &report)?;
    }
    if let Some(path) = &config.output_markdown {
        write_markdown(path, &report)?;
    }
    Ok(report)
}

fn write_json(path: &PathBuf, report: &DistillEvalReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = fs::File::create(path)?;
    serde_json::to_writer_pretty(file, report).map_err(AarambhError::Json)
}

fn write_markdown(path: &PathBuf, report: &DistillEvalReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    writeln!(file, "# Distillation Evaluation")?;
    writeln!(file)?;
    writeln!(file, "| Metric | Value |")?;
    writeln!(file, "|---|---:|")?;
    writeln!(file, "| Teacher backend | {} |", report.teacher_backend)?;
    writeln!(file, "| Prompts | {} |", report.prompts)?;
    writeln!(file, "| Rollouts | {} |", report.rollouts)?;
    writeln!(file, "| Completion tokens | {} |", report.completion_tokens)?;
    writeln!(file, "| Mean teacher reward | {:.6} |", report.reward_mean)?;
    if let Some(kl) = report.teacher_student_kl {
        writeln!(file, "| Teacher-student KL | {kl:.6} |")?;
    }
    writeln!(file, "| Tokens/s | {:.2} |", report.tokens_per_second)?;
    Ok(())
}

fn mean(values: &[f32]) -> f64 {
    values.iter().map(|value| *value as f64).sum::<f64>() / values.len().max(1) as f64
}

fn stddev(values: &[f32], mean: f32) -> f64 {
    (values
        .iter()
        .map(|value| (*value - mean).powi(2) as f64)
        .sum::<f64>()
        / values.len().max(1) as f64)
        .sqrt()
}
