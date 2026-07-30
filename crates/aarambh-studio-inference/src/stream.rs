use crate::engine::{FinishReason, GenerationStep};

#[derive(Debug, Clone)]
/// Streaming generation event.
pub enum StreamEvent {
    /// A token was produced.
    Token(GenerationStep),
    /// Generation finished.
    Finished(FinishReason),
}
