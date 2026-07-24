use std::path::PathBuf;
use std::str::FromStr;

use aarambh_ai_core::{AarambhError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
/// Runtime mode for the self-learning loop.
pub enum SelfLearnMode {
    /// CPU mode defers expensive updates by default.
    Cpu,
    /// GPU mode applies online updates inline.
    Gpu,
    /// Self-learning is disabled.
    Disabled,
}

impl SelfLearnMode {
    /// Return true when self-learning is enabled.
    pub fn is_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }
}

impl FromStr for SelfLearnMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            "disabled" | "none" | "off" => Ok(Self::Disabled),
            other => Err(format!(
                "invalid self-learning mode '{other}', expected disabled|cpu|gpu"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Online GRPO configuration used by self-learning.
pub struct OnlineGrpoConfig {
    /// Number of completions sampled for reward grouping.
    pub n_completions: usize,
    /// Sampling temperature.
    pub temperature: f32,
    /// Online adapter learning rate.
    pub online_lr: f64,
    /// KL penalty coefficient.
    pub kl_coeff: f64,
    /// LoRA adapter rank.
    pub lora_rank: usize,
    /// Whether CPU mode should store gradients instead of stepping inline.
    pub skip_inline_on_cpu: bool,
    /// Maximum generated tokens for self-learning rollouts.
    pub max_new_tokens: usize,
    /// Optional top-k sampling limit.
    pub top_k: Option<usize>,
    /// Optional nucleus sampling probability mass.
    pub top_p: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Replay buffer configuration.
pub struct ReplayConfig {
    /// Maximum stored replay entries.
    pub capacity: usize,
    /// Minimum critique score required for storage.
    pub min_score: f32,
    /// Online step interval for replay fine-tuning.
    pub replay_every_n: usize,
    /// Replay fine-tuning batch size.
    pub batch_size: usize,
    /// Replay JSONL path.
    pub path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Critique and rewrite configuration.
pub struct CritiqueConfig {
    /// Whether critique is enabled.
    pub enabled: bool,
    /// Score threshold below which rewrite is attempted.
    pub rewrite_threshold: f32,
    /// Maximum rewrite attempts.
    pub max_rewrites: usize,
    /// Maximum critique generation tokens.
    pub max_tokens: usize,
    #[serde(default = "default_rewrite_max_tokens")]
    /// Maximum rewrite generation tokens.
    pub rewrite_max_tokens: usize,
    /// Critique prompt template.
    pub prompt_template: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
/// Phase 38 capability probes attached to online parameter updates.
pub struct SelfLearnForgettingConfig {
    /// Enable forgetting diagnostics.
    pub enabled: bool,
    /// Capability probe manifest.
    pub manifest: PathBuf,
    /// Existing evaluation dataset root.
    pub data_dir: PathBuf,
    /// Training/evaluation config used by multimodal probes.
    pub config_path: Option<PathBuf>,
    /// Persistent forgetting-curve store.
    pub store: PathBuf,
    /// Optional seven-field JSONL export.
    pub jsonl: Option<PathBuf>,
    /// Optional per-task example cap.
    pub max_examples: Option<usize>,
    /// Generation budget for generative probes.
    pub max_new_tokens: usize,
    /// Tool-chain call budget.
    pub agent_max_steps: usize,
    /// Absolute score-change significance threshold.
    pub significance_threshold: f64,
    /// Permit HumanEval-lite subprocess execution.
    pub allow_code_exec: bool,
    /// Fail instead of recording unavailable probes.
    pub require_all_probes: bool,
    /// Optional stable baseline session identifier.
    pub baseline_id: Option<String>,
}

impl Default for SelfLearnForgettingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            manifest: PathBuf::from("data/eval/forgetting/probes.json"),
            data_dir: PathBuf::from("data/eval"),
            config_path: None,
            store: PathBuf::from("adapters/selflearn/forgetting_curves.json"),
            jsonl: None,
            max_examples: Some(8),
            max_new_tokens: 32,
            agent_max_steps: 8,
            significance_threshold: 0.02,
            allow_code_exec: false,
            require_all_probes: false,
            baseline_id: None,
        }
    }
}

impl SelfLearnForgettingConfig {
    /// Validate paths, limits, and score threshold.
    pub fn validate(&self) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        if self.manifest.as_os_str().is_empty() || self.store.as_os_str().is_empty() {
            return Err(AarambhError::Config(
                "self-learn forgetting manifest and store paths must be non-empty".into(),
            ));
        }
        if self
            .config_path
            .as_ref()
            .is_some_and(|path| path.as_os_str().is_empty())
        {
            return Err(AarambhError::Config(
                "self-learn forgetting config_path must not be empty".into(),
            ));
        }
        if self.max_examples == Some(0) || self.max_new_tokens == 0 || self.agent_max_steps == 0 {
            return Err(AarambhError::Config(
                "self-learn forgetting eval limits must be non-zero".into(),
            ));
        }
        if !self.significance_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.significance_threshold)
        {
            return Err(AarambhError::Config(
                "self-learn forgetting threshold must be finite and in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Complete self-learning configuration.
pub struct SelfLearnConfig {
    /// Runtime mode.
    pub mode: SelfLearnMode,
    /// Online GRPO settings.
    pub grpo: OnlineGrpoConfig,
    /// Replay settings.
    pub replay: ReplayConfig,
    /// Critique settings.
    pub critique: CritiqueConfig,
    /// Directory for adapter, optimizer, and metrics state.
    pub state_dir: PathBuf,
    /// Optional Phase 38 forgetting diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forgetting: Option<SelfLearnForgettingConfig>,
}

impl SelfLearnConfig {
    /// Return CPU-oriented defaults.
    pub fn for_cpu() -> Self {
        Self {
            mode: SelfLearnMode::Cpu,
            grpo: OnlineGrpoConfig {
                n_completions: 2,
                temperature: 0.8,
                online_lr: 1e-5,
                kl_coeff: 0.01,
                lora_rank: 8,
                skip_inline_on_cpu: true,
                max_new_tokens: 128,
                top_k: Some(50),
                top_p: Some(0.95),
            },
            replay: ReplayConfig {
                capacity: 500,
                min_score: 0.70,
                replay_every_n: 500,
                batch_size: 32,
                path: PathBuf::from("data/replay.jsonl"),
            },
            critique: CritiqueConfig {
                enabled: true,
                rewrite_threshold: 0.70,
                max_rewrites: 1,
                max_tokens: 50,
                rewrite_max_tokens: 32,
                prompt_template: default_critique_template(),
            },
            state_dir: PathBuf::from("adapters/selflearn"),
            forgetting: None,
        }
    }

    /// Return GPU-oriented defaults.
    pub fn for_gpu() -> Self {
        Self {
            mode: SelfLearnMode::Gpu,
            grpo: OnlineGrpoConfig {
                n_completions: 8,
                temperature: 0.8,
                online_lr: 1e-5,
                kl_coeff: 0.01,
                lora_rank: 16,
                skip_inline_on_cpu: false,
                max_new_tokens: 128,
                top_k: Some(50),
                top_p: Some(0.95),
            },
            replay: ReplayConfig {
                capacity: 5_000,
                min_score: 0.70,
                replay_every_n: 50,
                batch_size: 128,
                path: PathBuf::from("data/replay.jsonl"),
            },
            critique: CritiqueConfig {
                enabled: true,
                rewrite_threshold: 0.70,
                max_rewrites: 3,
                max_tokens: 50,
                rewrite_max_tokens: 64,
                prompt_template: default_critique_template(),
            },
            state_dir: PathBuf::from("adapters/selflearn"),
            forgetting: None,
        }
    }

    /// Return disabled self-learning defaults.
    pub fn disabled() -> Self {
        let mut config = Self::for_cpu();
        config.mode = SelfLearnMode::Disabled;
        config.critique.enabled = false;
        config
    }

    /// Return defaults for a mode.
    pub fn for_mode(mode: SelfLearnMode) -> Self {
        match mode {
            SelfLearnMode::Cpu => Self::for_cpu(),
            SelfLearnMode::Gpu => Self::for_gpu(),
            SelfLearnMode::Disabled => Self::disabled(),
        }
    }

    /// Override the replay JSONL path.
    pub fn with_replay_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.replay.path = path.into();
        self
    }

    /// Override the state directory.
    pub fn with_state_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.state_dir = path.into();
        self
    }

    /// Attach Phase 38 forgetting diagnostics.
    pub fn with_forgetting(mut self, config: SelfLearnForgettingConfig) -> Self {
        self.forgetting = Some(config);
        self
    }

    /// Validate configuration ranges.
    pub fn validate(&self) -> Result<()> {
        if self.grpo.n_completions == 0 {
            return Err(AarambhError::Config(
                "self-learn n_completions must be greater than zero".into(),
            ));
        }
        if self.grpo.online_lr <= 0.0 {
            return Err(AarambhError::Config(
                "self-learn online_lr must be positive".into(),
            ));
        }
        if self.grpo.kl_coeff < 0.0 {
            return Err(AarambhError::Config(
                "self-learn kl_coeff must be non-negative".into(),
            ));
        }
        if self.grpo.lora_rank == 0 {
            return Err(AarambhError::Config(
                "self-learn lora_rank must be greater than zero".into(),
            ));
        }
        if self.replay.capacity == 0 {
            return Err(AarambhError::Config(
                "self-learn replay capacity must be greater than zero".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.replay.min_score) {
            return Err(AarambhError::Config(
                "self-learn replay min_score must be in [0, 1]".into(),
            ));
        }
        if self.replay.replay_every_n == 0 {
            return Err(AarambhError::Config(
                "self-learn replay_every_n must be greater than zero".into(),
            ));
        }
        if self.replay.batch_size == 0 {
            return Err(AarambhError::Config(
                "self-learn replay batch_size must be greater than zero".into(),
            ));
        }
        if !(0.0..=1.0).contains(&self.critique.rewrite_threshold) {
            return Err(AarambhError::Config(
                "self-learn rewrite_threshold must be in [0, 1]".into(),
            ));
        }
        if self.critique.max_tokens == 0 {
            return Err(AarambhError::Config(
                "self-learn critique max_tokens must be greater than zero".into(),
            ));
        }
        if self.critique.rewrite_max_tokens == 0 {
            return Err(AarambhError::Config(
                "self-learn critique rewrite_max_tokens must be greater than zero".into(),
            ));
        }
        if let Some(forgetting) = &self.forgetting {
            forgetting.validate()?;
        }
        Ok(())
    }
}

impl Default for SelfLearnConfig {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Return the default JSON critique prompt template.
pub fn default_critique_template() -> String {
    r#"<|user|>
Rate this response on a scale from 0.0 to 1.0.
Score based on: accuracy, clarity, completeness, reasoning quality.

Question: {prompt}
Response: {response}

Reply with ONLY valid JSON and nothing else:
{"score": <float 0.0-1.0>, "reason": "<one sentence>"}
<|assistant|>
"#
    .to_string()
}

fn default_rewrite_max_tokens() -> usize {
    32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_defaults_match_i3_plan() {
        let config = SelfLearnConfig::for_cpu();
        assert_eq!(config.mode, SelfLearnMode::Cpu);
        assert_eq!(config.grpo.n_completions, 2);
        assert_eq!(config.grpo.lora_rank, 8);
        assert!(config.grpo.skip_inline_on_cpu);
        assert_eq!(config.replay.capacity, 500);
        assert_eq!(config.critique.max_rewrites, 1);
    }

    #[test]
    fn gpu_defaults_match_kaggle_plan() {
        let config = SelfLearnConfig::for_gpu();
        assert_eq!(config.mode, SelfLearnMode::Gpu);
        assert_eq!(config.grpo.n_completions, 8);
        assert_eq!(config.grpo.lora_rank, 16);
        assert!(!config.grpo.skip_inline_on_cpu);
        assert_eq!(config.replay.capacity, 5_000);
        assert_eq!(config.critique.max_rewrites, 3);
    }

    #[test]
    fn parses_disabled_aliases() {
        assert_eq!(
            "none".parse::<SelfLearnMode>().unwrap(),
            SelfLearnMode::Disabled
        );
        assert_eq!(
            "off".parse::<SelfLearnMode>().unwrap(),
            SelfLearnMode::Disabled
        );
    }
}
