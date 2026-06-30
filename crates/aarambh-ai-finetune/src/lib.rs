//! Fine-tuning support.
//!
//! Phase 9 implements LoRA, QLoRA, SFT loss masking, and adapter merge support.
//! Phase 10 adds deterministic-verifier GRPO for adapter-only RL fine-tuning.
#![deny(missing_docs)]

/// Adapter metadata and serialization helpers.
pub mod adapter;
/// Group Relative Policy Optimization data loading, rollout, and training.
pub mod grpo;
/// LoRA adapter layers and configuration.
pub mod lora;
/// LoRA-wrapped Aarambh model implementation.
pub mod model;
/// Supervised fine-tuning datasets, templates, and batches.
pub mod sft;
/// SFT trainer and adapter merge entrypoints.
pub mod trainer;
/// Rule-based verifiers used by GRPO and self-learning.
pub mod verifier;

pub use adapter::{AdapterMetadata, load_adapter_metadata, save_adapter};
pub use grpo::{
    GrpoConfig, GrpoDataset, GrpoExample, GrpoMetrics, GrpoRunConfig, GrpoThinkingMode,
    GrpoTrainer, Rollout, RolloutFinish, compute_advantages, grpo_loss, run_grpo_from_config,
    sample_group,
};
pub use lora::{BaseLinear, LoraConfig, LoraLinear};
pub use model::LoraAarambhModel;
pub use sft::{
    ChatTemplate, SftBatch, SftDataLoader, SftDataset, SftExample, ThinkingSftExample,
    format_thinking_sft,
};
pub use trainer::{SftRunConfig, SftTrainer, merge_lora_from_paths, run_sft_from_config};
pub use verifier::{
    CompositeVerifier, FormatVerifier, MathVerifier, Verifier, VerifierKind, extract_final_number,
};
