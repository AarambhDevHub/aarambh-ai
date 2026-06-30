//! Transformer model components and the full Aarambh causal language model.
#![deny(missing_docs)]

/// Token embedding layer.
pub mod embedding;
/// Language-model projection head.
pub mod head;
/// Full decoder-only Aarambh model.
pub mod model;

pub use embedding::TokenEmbedding;
pub use head::LmHead;
pub use model::AarambhModel;
