use std::path::PathBuf;

use aarambh_ai_core::{AarambhError, ModelConfig, Result};
use aarambh_ai_finetune::{SftExample, Verifier};
use aarambh_ai_inference::{GenerationConfig, GenerationOutput, GenerationStep};
use candle_core::Device as CandleDevice;

use crate::config::SelfLearnConfig;
use crate::critique::{CritiqueResult, critique_response};
use crate::metrics::{LearningMetrics, MetricsEvent};
use crate::online_grpo::{OnlineGrpo, OnlineGrpoBuildConfig, OnlineUpdate};
use crate::replay::{ReplayBuffer, ReplayEntry};

#[derive(Debug, Clone)]
pub struct SelfLearnBuildConfig {
    pub model_config: ModelConfig,
    pub base_model_path: PathBuf,
    pub reference_model_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub config: SelfLearnConfig,
    pub device: CandleDevice,
    pub seed: u64,
}

#[derive(Debug)]
pub struct SelfLearnDraft {
    pub prompt: String,
    pub output: GenerationOutput,
    pub update: OnlineUpdate,
    pub critique: CritiqueResult,
}

#[derive(Debug, Clone)]
pub struct SelfLearnResponse {
    pub response: String,
    pub critique_score: f32,
    pub verifier_score: Option<f32>,
    pub was_rewritten: bool,
    pub stored_in_replay: bool,
    pub used_grpo: bool,
    pub metrics_summary: String,
}

pub struct SelfLearnLoop {
    online_grpo: OnlineGrpo,
    replay: ReplayBuffer,
    metrics: LearningMetrics,
    config: SelfLearnConfig,
    last_draft: Option<SelfLearnDraft>,
}

impl SelfLearnLoop {
    pub fn from_paths(build: SelfLearnBuildConfig) -> Result<Self> {
        build.config.validate()?;
        let replay =
            ReplayBuffer::load_jsonl(&build.config.replay.path, build.config.replay.clone())?;
        let metrics = LearningMetrics::load_jsonl(build.config.state_dir.join("metrics.jsonl"))?;
        let online_grpo = OnlineGrpo::from_paths(OnlineGrpoBuildConfig {
            model_config: build.model_config,
            base_model_path: build.base_model_path,
            reference_model_path: build.reference_model_path,
            tokenizer_path: build.tokenizer_path,
            state_dir: build.config.state_dir.clone(),
            config: build.config.grpo.clone(),
            mode: build.config.mode,
            device: build.device,
            seed: build.seed,
        })?;
        Ok(Self {
            online_grpo,
            replay,
            metrics,
            config: build.config,
            last_draft: None,
        })
    }

    pub fn config(&self) -> &SelfLearnConfig {
        &self.config
    }

    pub fn replay(&self) -> &ReplayBuffer {
        &self.replay
    }

    pub fn metrics(&self) -> &LearningMetrics {
        &self.metrics
    }

    pub fn generate_draft(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        verifier: Option<&dyn Verifier>,
        ground_truth: Option<&str>,
    ) -> Result<&SelfLearnDraft> {
        let update = self
            .online_grpo
            .generate_update(prompt, config, verifier, ground_truth)?;
        let base_response = update.output.text.clone();
        let critique = if self.config.critique.enabled {
            critique_response(
                &mut self.online_grpo,
                prompt,
                &base_response,
                &self.config.critique,
            )?
        } else {
            CritiqueResult {
                response: base_response,
                score: 0.5,
                reason: "critique disabled".into(),
                was_rewritten: false,
            }
        };
        let mut output = update.output.clone();
        if critique.was_rewritten {
            output.text = critique.response.clone();
            output.answer_text = critique.response.clone();
            output.raw_text = critique.response.clone();
        }
        self.last_draft = Some(SelfLearnDraft {
            prompt: prompt.to_string(),
            output,
            update,
            critique,
        });
        Ok(self.last_draft.as_ref().expect("draft inserted above"))
    }

    pub fn generate_draft_with_callback<F>(
        &mut self,
        prompt: &str,
        config: GenerationConfig,
        verifier: Option<&dyn Verifier>,
        ground_truth: Option<&str>,
        mut on_step: F,
    ) -> Result<GenerationOutput>
    where
        F: FnMut(&GenerationStep) -> Result<()>,
    {
        let output = self
            .generate_draft(prompt, config, verifier, ground_truth)?
            .output
            .clone();
        for step in &output.steps {
            on_step(step)?;
        }
        Ok(output)
    }

    pub fn commit_last_draft(
        &mut self,
        safe_response: Option<String>,
    ) -> Result<SelfLearnResponse> {
        let mut draft = self
            .last_draft
            .take()
            .ok_or_else(|| AarambhError::Config("no self-learning draft to commit".into()))?;
        if let Some(response) = safe_response {
            draft.output.text = response.clone();
            draft.output.answer_text = response.clone();
            draft.output.raw_text = response;
        }
        let score = draft.critique.score;
        let mut stored = false;
        if score >= self.config.replay.min_score {
            let entry = ReplayEntry::new(&draft.prompt, &draft.output.text, score);
            stored = self.replay.push(entry.clone());
            if stored {
                ReplayBuffer::append_jsonl(&self.config.replay.path, &entry)?;
            }
        }
        let verifier_score = draft.update.verifier_score;
        let used_grpo = draft.update.used_grpo;
        let _ = self.online_grpo.commit_update(draft.update)?;
        self.metrics.record(score, &draft.prompt);
        if self.replay.should_replay(self.online_grpo.step_count()) {
            let _ = self.replay_finetune()?;
        }
        let metrics_event = MetricsEvent {
            step: self.metrics.total_steps(),
            topic: crate::replay::infer_topic(&draft.prompt),
            score,
        };
        LearningMetrics::append_event(self.config.state_dir.join("metrics.jsonl"), &metrics_event)?;
        Ok(SelfLearnResponse {
            response: draft.output.text,
            critique_score: score,
            verifier_score,
            was_rewritten: draft.critique.was_rewritten,
            stored_in_replay: stored,
            used_grpo,
            metrics_summary: self.metrics.summary(),
        })
    }

    pub fn discard_last_draft(&mut self) {
        self.last_draft = None;
    }

    pub fn flush_pending_gradients(&mut self) -> Result<Option<f64>> {
        self.online_grpo.flush_pending_gradients()
    }

    pub fn replay_finetune(&mut self) -> Result<Option<f64>> {
        let batch = self.replay.sample_batch(self.config.replay.batch_size);
        if batch.is_empty() {
            return Ok(None);
        }
        let path = self.config.state_dir.join("replay_sft.jsonl");
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(&path)?;
        let mut examples = Vec::with_capacity(batch.len());
        for entry in &batch {
            let example = SftExample {
                instruction: entry.prompt.clone(),
                input: None,
                response: entry.response.clone(),
            };
            serde_json::to_writer(&mut file, &example)?;
            use std::io::Write;
            writeln!(file)?;
            examples.push(example);
        }
        let grad_norm = self
            .online_grpo
            .replay_sft_batch(&examples, self.config.replay.batch_size)?;
        self.metrics.record_replay();
        Ok(grad_norm)
    }
}
