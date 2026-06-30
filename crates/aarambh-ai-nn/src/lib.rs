//! Neural-network layers used by the Aarambh transformer stack.
#![deny(missing_docs)]

/// Grouped-query attention layer.
pub mod attention;
/// Transformer block composition.
pub mod block;
/// SwiGLU feed-forward layer.
pub mod ffn;
/// Training/inference KV cache helper.
pub mod kvcache;
/// RMSNorm layer.
pub mod norm;
/// Rotary-position embedding cache.
pub mod rope;

pub use attention::GroupedQueryAttention;
pub use block::TransformerBlock;
pub use ffn::SwiGluFfn;
pub use kvcache::KVCache;
pub use norm::RMSNorm;
pub use rope::RopeCache;
