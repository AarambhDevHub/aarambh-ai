//! Local OpenAI-compatible HTTP/SSE inference server.
#![deny(missing_docs)]

/// OpenAI-compatible HTTP data types.
pub mod api;
/// Continuous inference scheduling and request sessions.
pub mod batching;
/// Atomic server telemetry.
pub mod metrics;
/// Axum routing and server lifecycle.
pub mod server;

pub use batching::{BatcherConfig, BatcherHandle, GenerationEvent, GenerationRequest};
pub use metrics::{MetricsSnapshot, ServerMetrics};
pub use server::{ServeConfig, build_router, run_server};
