use thiserror::Error;

#[derive(Debug, Error)]
/// Error type shared by Aarambh AI crates.
pub enum AarambhError {
    /// Configuration parsing or validation failed.
    #[error("Configuration error: {0}")]
    Config(String),
    /// Tensor or model shape validation failed.
    #[error("Shape mismatch: {0}")]
    Shape(String),
    /// Candle runtime returned an error.
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    /// File-system IO failed.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON serialization or parsing failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// Tokenizer loading, encoding, or decoding failed.
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    /// Checkpoint loading or saving failed.
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),
    /// The requested operation is unsupported by this build or input.
    #[error("Unsupported operation: {0}")]
    Unsupported(String),
}

/// Result alias using [`AarambhError`].
pub type Result<T> = std::result::Result<T, AarambhError>;
