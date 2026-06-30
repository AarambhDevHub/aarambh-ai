//! Dataset loading, batching, and preprocessing utilities for training runs.
#![deny(missing_docs)]

/// Text dataset abstractions and file-backed datasets.
pub mod dataset;
/// Deterministic mini-batch loader for tokenized language-model data.
pub mod loader;
/// Tokenization and fixed-window chunking helpers.
pub mod preprocess;

pub use dataset::{JsonlDataset, PlaintextDataset, TextDataset};
pub use loader::{Batch, DataLoader};
pub use preprocess::chunk_and_tokenize;
