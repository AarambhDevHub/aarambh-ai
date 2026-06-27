use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub ffn_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub norm_eps: f64,
    pub tie_embeddings: bool,
}

impl ModelConfig {
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

    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.n_heads
    }

    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub lr: f64,
    pub batch_size: usize,
    pub grad_accum_steps: usize,
    pub max_epochs: usize,
    pub max_steps: usize,
    pub warmup_steps: usize,
    pub min_lr_ratio: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub clip_grad_norm: f64,
    pub save_every_n_steps: usize,
    pub log_every_n_steps: usize,
    pub eval_steps: usize,
    pub seed: u64,
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
