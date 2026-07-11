//! Autoregressive inference engine, sampling, streaming, KV cache, and thinking controls.
#![deny(missing_docs)]

/// Generation engine and output types.
pub mod engine;
/// Grammar-constrained JSON decoding.
pub mod grammar;
/// Inference-time key/value cache.
pub mod kvcache;
/// Temperature, top-k, top-p, and greedy sampling.
pub mod sampler;
/// Exact draft-model speculative decoding.
pub mod speculative;
/// Streaming callback event types.
pub mod stream;
/// Thinking budget and forced-token controls.
pub mod thinking;
/// Tool definitions, call protocol, and decoding controller.
pub mod tool_calling;

pub use engine::{
    FinishReason, GenerationConfig, GenerationOutput, GenerationPhase, GenerationStep,
    InferenceEngine,
};
pub use grammar::{JsonSchema, JsonSchemaGrammar};
pub use kvcache::KvCache;
pub use sampler::{Sampler, TokenCandidate};
pub use speculative::{SpeculativeConfig, SpeculativeEngine, SpeculativeStats};
pub use stream::StreamEvent;
pub use thinking::{ForceToken, ThinkingController, ThinkingMode};
pub use tool_calling::{
    FINAL_MARKER, TOOL_CALL_END, TOOL_CALL_START, ToolCall, ToolCallController, ToolCallingConfig,
    ToolChoice, ToolDefinition,
};
