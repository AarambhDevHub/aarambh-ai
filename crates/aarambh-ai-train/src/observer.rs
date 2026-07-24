use aarambh_ai_core::Result;
use aarambh_ai_model::AarambhModel;
use candle_core::{DType, Device};

/// Point in a training run at which a read-only observer is invoked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingObserverEvent {
    /// Model state immediately before training continues.
    Start,
    /// Model state immediately after an optimizer update.
    OptimizerStep,
    /// Final model state before the last checkpoint is written.
    Finish,
}

/// Read-only live model state supplied to a training observer.
pub struct TrainingObserverSnapshot<'a> {
    /// Observer event.
    pub event: TrainingObserverEvent,
    /// Current optimizer step.
    pub step: usize,
    /// Live training model.
    pub model: &'a AarambhModel,
    /// Model device.
    pub device: &'a Device,
    /// Model dtype.
    pub dtype: DType,
}

/// Dependency-neutral hook for diagnostics that inspect live training state.
pub trait TrainingObserver {
    /// Return whether this event should pause all ranks and invoke the observer.
    fn should_observe(&self, event: TrainingObserverEvent, step: usize) -> bool;

    /// Inspect a rank-0 model snapshot without mutating training state.
    fn observe(&mut self, snapshot: TrainingObserverSnapshot<'_>) -> Result<()>;
}
