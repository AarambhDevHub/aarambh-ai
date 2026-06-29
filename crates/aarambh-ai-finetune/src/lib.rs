//! Fine-tuning support.
//!
//! Phase 9 implements LoRA, QLoRA, SFT loss masking, and adapter merge support.

pub mod adapter;
pub mod lora;
pub mod model;
pub mod sft;
pub mod trainer;

pub use adapter::{AdapterMetadata, load_adapter_metadata, save_adapter};
pub use lora::{BaseLinear, LoraConfig, LoraLinear};
pub use model::LoraAarambhModel;
pub use sft::{
    ChatTemplate, SftBatch, SftDataLoader, SftDataset, SftExample, ThinkingSftExample,
    format_thinking_sft,
};
pub use trainer::{SftRunConfig, SftTrainer, merge_lora_from_paths, run_sft_from_config};
