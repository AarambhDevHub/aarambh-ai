use crate::engine::{FinishReason, GenerationStep};

#[derive(Debug, Clone)]
pub enum StreamEvent {
    Token(GenerationStep),
    Finished(FinishReason),
}
