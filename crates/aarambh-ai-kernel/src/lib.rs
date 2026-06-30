//! Kernel dispatch and optimized CPU/CUDA kernels for aarambh-ai.
//!
//! CPU builds use SIMD/parallel kernels where available and fall back to Candle
//! for unsupported devices, dtypes, or layouts. CUDA builds with NVCC available
//! additionally load Phase 14 PTX kernels for Flash Attention and fused layer
//! primitives.
#![deny(missing_docs)]

/// CPU kernel implementations.
pub mod cpu;
/// Runtime dispatch between Candle fallback, CPU kernels, and CUDA kernels.
pub mod dispatch;
/// Flash Attention kernel entrypoints.
pub mod flash_attn;
/// Fused feed-forward kernel entrypoints.
pub mod fused_ffn;
/// Fused RMSNorm kernel entrypoints.
pub mod fused_norm;
/// Fused RoPE kernel entrypoints.
pub mod fused_rope;

pub use dispatch::KernelPath;
pub use dispatch::attention_forward;
pub use dispatch::attention_forward_candle;
pub use dispatch::attention_forward_train;
pub use dispatch::attention_path;
pub use dispatch::rms_norm;
pub use dispatch::rms_norm_path;
