//! Evaluation harness for perplexity, multiple-choice, math, and code tasks.
#![deny(missing_docs)]

/// Greedy generation helpers used by generative eval tasks.
pub mod generation;
/// Shared task runner and eval context types.
pub mod harness;
/// Perplexity-on-holdout evaluation.
pub mod ppl;
/// Scorecard serialization and comparison.
pub mod report;
/// Continuation log-probability scoring helpers.
pub mod scoring;
/// Built-in evaluation task implementations.
pub mod tasks;

pub use generation::greedy_generate;
pub use harness::{EvalConfig, EvalContext, EvalTask, run_all};
pub use ppl::{PplResult, compute_ppl};
pub use report::{ScoreDelta, Scorecard, ScorecardComparison, TaskScore};
pub use scoring::{ContinuationScore, ContinuationScorer, ModelLogProbScorer};
