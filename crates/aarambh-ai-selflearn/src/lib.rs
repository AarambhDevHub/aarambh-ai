//! Self-learning replay, critique, metrics, and online adapter update loop.
#![deny(missing_docs)]

/// Self-learning configuration types.
pub mod config;
/// Critique parsing and scoring helpers.
pub mod critique;
/// High-level self-learning orchestration.
pub mod learning_loop;
/// Learning metric tracking.
pub mod metrics;
/// Online GRPO adapter update implementation.
pub mod online_grpo;
/// Replay buffer persistence and sampling.
pub mod replay;

pub use config::{CritiqueConfig, OnlineGrpoConfig, ReplayConfig, SelfLearnConfig, SelfLearnMode};
pub use critique::{CritiqueResult, critique_response, parse_critique_response};
pub use learning_loop::{SelfLearnBuildConfig, SelfLearnDraft, SelfLearnLoop, SelfLearnResponse};
pub use metrics::{LearningMetrics, TrendDirection};
pub use online_grpo::{OnlineGrpo, OnlineGrpoBuildConfig, OnlineUpdate, generate_lora};
pub use replay::{ReplayBuffer, ReplayEntry, ReplayStats, infer_topic};
