//! Fine-tuning support.
//!
//! Full LoRA, QLoRA, SFT, GRPO, and verifier implementations come in Phase 9.

pub mod sft;

pub use sft::{ThinkingSftExample, format_thinking_sft};
