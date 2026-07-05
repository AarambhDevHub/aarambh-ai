use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::{AarambhError, Result};

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// RoPE scaling strategy used to extend context length.
pub enum RopeScalingMethod {
    /// YaRN interpolation with beta correction ramp.
    #[default]
    Yarn,
    /// NTK-aware theta rescaling.
    Ntk,
    /// Linear inverse-frequency scaling.
    Linear,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
/// Configuration for long-context RoPE scaling.
pub struct RopeScalingConfig {
    /// Scaling method to apply.
    pub method: RopeScalingMethod,
    /// Context extension factor relative to the original context length.
    pub factor: f64,
    /// Context length used during the base model's original training.
    pub original_max_seq_len: usize,
    /// YaRN high-frequency correction boundary.
    pub beta_fast: f64,
    /// YaRN low-frequency correction boundary.
    pub beta_slow: f64,
    /// Multiplicative YaRN attention scale applied to cos/sin tables.
    pub attn_factor: f64,
}

impl Default for RopeScalingConfig {
    fn default() -> Self {
        Self {
            method: RopeScalingMethod::Yarn,
            factor: 1.0,
            original_max_seq_len: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attn_factor: 1.0,
        }
    }
}

impl RopeScalingConfig {
    /// Validate this scaling config against a model context length and RoPE head dimension.
    pub fn validate(&self, max_seq_len: usize, head_dim: usize) -> Result<()> {
        if self.factor <= 1.0 || !self.factor.is_finite() {
            return Err(AarambhError::Config(
                "rope_scaling.factor must be finite and greater than 1.0".into(),
            ));
        }
        if self.original_max_seq_len == 0 {
            return Err(AarambhError::Config(
                "rope_scaling.original_max_seq_len must be non-zero".into(),
            ));
        }
        if max_seq_len < self.original_max_seq_len {
            return Err(AarambhError::Config(format!(
                "max_seq_len {max_seq_len} must be >= rope_scaling.original_max_seq_len {}",
                self.original_max_seq_len
            )));
        }
        if head_dim <= 2 {
            return Err(AarambhError::Config(
                "head_dim must be greater than 2 for RoPE scaling".into(),
            ));
        }
        if self.attn_factor <= 0.0 || !self.attn_factor.is_finite() {
            return Err(AarambhError::Config(
                "rope_scaling.attn_factor must be finite and positive".into(),
            ));
        }
        if matches!(self.method, RopeScalingMethod::Yarn) {
            if self.beta_fast <= 0.0
                || self.beta_slow <= 0.0
                || !self.beta_fast.is_finite()
                || !self.beta_slow.is_finite()
            {
                return Err(AarambhError::Config(
                    "rope_scaling beta values must be finite and positive".into(),
                ));
            }
            if self.beta_fast <= self.beta_slow {
                return Err(AarambhError::Config(
                    "rope_scaling.beta_fast must be greater than beta_slow".into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Decoder-only transformer model shape and numerical defaults.
pub struct ModelConfig {
    /// Number of tokenizer entries supported by the model.
    pub vocab_size: usize,
    /// Width of token embeddings and hidden states.
    pub hidden_dim: usize,
    /// Intermediate width of the feed-forward network.
    pub ffn_dim: usize,
    /// Number of transformer decoder blocks.
    pub n_layers: usize,
    /// Number of query attention heads.
    pub n_heads: usize,
    /// Number of key/value heads used by grouped-query attention.
    pub n_kv_heads: usize,
    /// Maximum context length in tokens.
    pub max_seq_len: usize,
    /// Rotary-position embedding base frequency.
    pub rope_theta: f64,
    /// Optional long-context RoPE scaling configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
    /// Whether the output head shares weights with token embeddings.
    pub tie_embeddings: bool,
}

impl ModelConfig {
    /// Return the tiny smoke-test model preset.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 384,
            ffn_dim: 1024,
            n_layers: 8,
            n_heads: 6,
            n_kv_heads: 2,
            max_seq_len: 512,
            rope_theta: 10000.0,
            rope_scaling: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Return the small v1 training preset.
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 768,
            ffn_dim: 2688,
            n_layers: 12,
            n_heads: 12,
            n_kv_heads: 4,
            max_seq_len: 1024,
            rope_theta: 10000.0,
            rope_scaling: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Return the medium scale-up preset.
    pub fn medium() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 1024,
            ffn_dim: 3392,
            n_layers: 24,
            n_heads: 16,
            n_kv_heads: 8,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rope_scaling: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Return the large v1 target preset.
    pub fn large() -> Self {
        Self {
            vocab_size: 32000,
            hidden_dim: 2048,
            ffn_dim: 6656,
            n_layers: 24,
            n_heads: 32,
            n_kv_heads: 8,
            max_seq_len: 4096,
            rope_theta: 500000.0,
            rope_scaling: None,
            norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Return the per-head hidden width.
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    /// Load model configuration from a JSON file.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_model_config_json_defaults_rope_scaling_to_none() {
        let json = r#"{
            "vocab_size": 32000,
            "hidden_dim": 384,
            "ffn_dim": 1024,
            "n_layers": 8,
            "n_heads": 6,
            "n_kv_heads": 2,
            "max_seq_len": 512,
            "rope_theta": 10000.0,
            "norm_eps": 0.00001,
            "tie_embeddings": true
        }"#;
        let cfg: ModelConfig = serde_json::from_str(json).unwrap();
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn rope_scaling_validation_rejects_invalid_factor() {
        let cfg = RopeScalingConfig {
            factor: 1.0,
            original_max_seq_len: 512,
            ..RopeScalingConfig::default()
        };
        let err = cfg.validate(1024, 64).unwrap_err().to_string();
        assert!(err.contains("factor"), "{err}");
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
/// Training hyperparameters and checkpoint cadence.
pub struct TrainConfig {
    /// Peak learning rate.
    pub lr: f64,
    /// Number of sequences per micro-batch.
    pub batch_size: usize,
    /// Number of micro-batches accumulated before an optimizer step.
    pub grad_accum_steps: usize,
    /// Maximum number of full dataset passes.
    pub max_epochs: usize,
    /// Maximum optimizer steps.
    pub max_steps: usize,
    /// Number of warmup optimizer steps.
    pub warmup_steps: usize,
    /// Final learning-rate ratio relative to the peak rate.
    pub min_lr_ratio: f64,
    /// AdamW decoupled weight decay.
    pub weight_decay: f64,
    /// AdamW first-moment coefficient.
    pub beta1: f64,
    /// AdamW second-moment coefficient.
    pub beta2: f64,
    /// AdamW numerical epsilon.
    pub epsilon: f64,
    /// Maximum global gradient norm.
    pub clip_grad_norm: f64,
    /// Checkpoint save interval in optimizer steps.
    pub save_every_n_steps: usize,
    /// Training log interval in optimizer steps.
    pub log_every_n_steps: usize,
    /// Evaluation interval in optimizer steps.
    pub eval_steps: usize,
    /// Random seed used by loaders and sampling.
    pub seed: u64,
    /// Directory where checkpoints are written.
    pub checkpoint_dir: std::path::PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            batch_size: 2,
            grad_accum_steps: 16,
            max_epochs: 1,
            max_steps: 5000,
            warmup_steps: 200,
            min_lr_ratio: 0.1,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.95,
            epsilon: 1e-8,
            clip_grad_norm: 1.0,
            save_every_n_steps: 1000,
            log_every_n_steps: 10,
            eval_steps: 500,
            seed: 42,
            checkpoint_dir: std::path::PathBuf::from("checkpoints"),
        }
    }
}
