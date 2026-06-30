use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::Result;

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
