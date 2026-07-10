//! Training configuration, trainer loop, optimizer, schedules, checkpoints, and loss helpers.
#![deny(missing_docs)]

/// Checkpoint save/load helpers.
pub mod checkpoint;
/// TOML-backed training run configuration.
pub mod config;
/// Single-node data-parallel training helpers.
pub mod distributed;
/// Language-model loss functions.
pub mod loss;
/// Optimizer and gradient utilities.
pub mod optim;
/// Learning-rate schedules.
pub mod schedule;
/// Main training loop.
pub mod trainer;
/// Vision-projector-only pretraining loop.
pub mod vision_projector;

pub use checkpoint::{CheckpointManager, TrainState};
pub use config::{TrainingRunConfig, run_training_from_config};
pub use distributed::{
    DistributedBackend, DistributedConfig, DistributedContext, DistributedRuntime,
    ResolvedDistributedConfig,
};
pub use loss::cross_entropy_loss;
pub use optim::{AdamW, AdamWConfig, GradMap, TrainableParameter};
pub use schedule::CosineScheduleWithWarmup;
pub use trainer::{Trainer, TrainingMetrics};
pub use vision_projector::{VisionTrainingConfig, run_projector_pretrain};
