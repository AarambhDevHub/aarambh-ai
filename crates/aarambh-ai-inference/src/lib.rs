//! Autoregressive inference engine, sampling, streaming, KV cache, and thinking controls.
#![deny(missing_docs)]

/// Generation engine and output types.
pub mod engine;
/// Inference-time key/value cache.
pub mod kvcache;
/// Temperature, top-k, top-p, and greedy sampling.
pub mod sampler;
/// Streaming callback event types.
pub mod stream;
/// Thinking budget and forced-token controls.
pub mod thinking;

pub use engine::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep,
    InferenceEngine,
};
pub use kvcache::KvCache;
pub use sampler::{Sampler, TokenCandidate};
pub use stream::StreamEvent;
pub use thinking::{ForceToken, ThinkingController, ThinkingMode};
