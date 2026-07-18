use std::str::FromStr;

use aarambh_ai_core::{AarambhError, Result};
use aarambh_ai_inference::ThinkingMode;
use serde::{Deserialize, Serialize};

/// Distillation objective applied to student-generated completions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DistillObjective {
    /// Forward KL from the frozen teacher distribution to the student.
    #[default]
    SoftKl,
    /// Group-normalized policy loss weighted by teacher rewards.
    Reward,
}

impl FromStr for DistillObjective {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "soft-kl" | "kl" => Ok(Self::SoftKl),
            "reward" => Ok(Self::Reward),
            other => Err(format!(
                "unsupported distillation objective '{other}', expected soft-kl or reward"
            )),
        }
    }
}

/// Thinking controller used while collecting student rollouts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DistillThinkingMode {
    /// Do not force a thinking block.
    #[default]
    None,
    /// Use the low thinking budget.
    Low,
    /// Use the medium thinking budget.
    Medium,
    /// Use the high thinking budget.
    High,
}

impl From<DistillThinkingMode> for ThinkingMode {
    fn from(value: DistillThinkingMode) -> Self {
        match value {
            DistillThinkingMode::None => Self::None,
            DistillThinkingMode::Low => Self::Low,
            DistillThinkingMode::Medium => Self::Medium,
            DistillThinkingMode::High => Self::High,
        }
    }
}

impl FromStr for DistillThinkingMode {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            other => Err(format!(
                "unsupported distillation thinking mode '{other}', expected none, low, medium, or high"
            )),
        }
    }
}

/// Rollout and objective settings for on-policy distillation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DistillConfig {
    /// Number of independently sampled student completions per prompt.
    pub rollouts_per_prompt: usize,
    /// Maximum sampled tokens per completion.
    pub max_new_tokens: usize,
    /// Student rollout sampling temperature.
    pub temperature: f32,
    /// Optional nucleus sampling threshold.
    pub top_p: Option<f32>,
    /// Optional top-k sampling limit.
    pub top_k: Option<usize>,
    /// Temperature applied to local-teacher soft targets.
    pub teacher_temperature: f64,
    /// Absolute bound applied to normalized reward advantages.
    pub advantage_clip: f64,
    /// Thinking mode used for rollout generation.
    pub thinking: DistillThinkingMode,
    /// Distillation objective.
    pub objective: DistillObjective,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            rollouts_per_prompt: 4,
            max_new_tokens: 128,
            temperature: 0.8,
            top_p: Some(0.95),
            top_k: Some(50),
            teacher_temperature: 1.0,
            advantage_clip: 5.0,
            thinking: DistillThinkingMode::None,
            objective: DistillObjective::SoftKl,
        }
    }
}

impl DistillConfig {
    /// Validate rollout sampling and objective parameters.
    pub fn validate(&self) -> Result<()> {
        if self.rollouts_per_prompt == 0 {
            return Err(AarambhError::Config(
                "distill rollouts_per_prompt must be non-zero".into(),
            ));
        }
        if self.objective == DistillObjective::Reward && self.rollouts_per_prompt < 2 {
            return Err(AarambhError::Config(
                "reward distillation requires at least two rollouts per prompt".into(),
            ));
        }
        if self.max_new_tokens == 0 {
            return Err(AarambhError::Config(
                "distill max_new_tokens must be non-zero".into(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(AarambhError::Config(
                "distill temperature must be finite and non-negative".into(),
            ));
        }
        if let Some(top_p) = self.top_p
            && !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0)
        {
            return Err(AarambhError::Config(
                "distill top_p must be finite and in (0, 1]".into(),
            ));
        }
        if self.top_k == Some(0) {
            return Err(AarambhError::Config(
                "distill top_k must be non-zero when configured".into(),
            ));
        }
        if self.teacher_temperature <= 0.0 || !self.teacher_temperature.is_finite() {
            return Err(AarambhError::Config(
                "teacher_temperature must be finite and greater than zero".into(),
            ));
        }
        if self.advantage_clip <= 0.0 || !self.advantage_clip.is_finite() {
            return Err(AarambhError::Config(
                "advantage_clip must be finite and greater than zero".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid_for_soft_kl() {
        DistillConfig::default().validate().unwrap();
    }

    #[test]
    fn reward_objective_requires_a_rollout_group() {
        let config = DistillConfig {
            rollouts_per_prompt: 1,
            objective: DistillObjective::Reward,
            ..DistillConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn optional_sampling_filters_reject_zero_values() {
        let zero_top_p = DistillConfig {
            top_p: Some(0.0),
            ..DistillConfig::default()
        };
        assert!(zero_top_p.validate().is_err());
        let zero_top_k = DistillConfig {
            top_k: Some(0),
            ..DistillConfig::default()
        };
        assert!(zero_top_k.validate().is_err());
    }

    #[test]
    fn objective_and_thinking_parsers_are_strict() {
        assert_eq!(
            DistillObjective::from_str("kl").unwrap(),
            DistillObjective::SoftKl
        );
        assert_eq!(
            DistillThinkingMode::from_str("medium").unwrap(),
            DistillThinkingMode::Medium
        );
        assert!(DistillObjective::from_str("reverse-kl").is_err());
        assert!(DistillThinkingMode::from_str("max").is_err());
    }
}
