use thiserror::Error;

#[derive(Debug, Error)]
pub enum AarambhError {
    #[error("Configuration error: {0}")]
    Config(String),
    #[error("Shape mismatch: {0}")]
    Shape(String),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, AarambhError>;
