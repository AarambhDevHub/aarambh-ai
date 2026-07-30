//! On-policy distillation with local-checkpoint and scored-reference teachers.
#![deny(missing_docs)]

/// Full-model distillation checkpoint persistence.
pub mod checkpoint;
/// Distillation configuration and validated objective selectors.
pub mod config;
/// Prompt, scored-reference, offline, and replay-batch data structures.
pub mod dataset;
/// On-policy and offline distillation objectives.
pub mod distill_loss;
/// Held-out fresh-rollout distillation evaluation.
pub mod evaluate;
/// Static teacher-completion preparation and matched offline control training.
pub mod offline;
/// Student rollout generation through the inference engine.
pub mod rollout;
/// Frozen teacher scoring backends.
pub mod teacher_score;
/// Full-weight on-policy trainer and run configuration.
pub mod trainer;

pub use checkpoint::{DistillCheckpointManager, DistillState};
pub use config::{DistillConfig, DistillObjective, DistillThinkingMode};
pub use dataset::{
    OfflineDataset, OfflineExample, PromptDataset, PromptExample, ReferenceAnswer, ReplayBatch,
    ScoredReferenceDataset, ScoredReferenceRecord,
};
pub use distill_loss::{
    RewardLossOutput, SoftKlLossOutput, group_normalized_advantages, reward_policy_loss,
    soft_kl_loss,
};
pub use evaluate::{DistillEvalConfig, DistillEvalReport, evaluate_distillation};
pub use offline::{
    OfflinePrepareConfig, OfflineRunConfig, prepare_offline_dataset,
    run_offline_distill_from_config,
};
pub use rollout::{RolloutFinish, StudentRollout, generate_student_rollouts};
pub use teacher_score::{
    LocalCheckpointTeacher, ScoredDatasetTeacher, TeacherBatchFeedback, TeacherScore,
    TeacherScorer, TeacherSignal,
};
pub use trainer::{
    DistillMetrics, DistillRunConfig, DistillTrainer, TeacherSourceConfig, run_distill_from_config,
};
