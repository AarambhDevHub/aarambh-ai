//! Kernel dispatch and optimized CPU/CUDA kernels for aarambh-ai.
//!
//! CPU builds use SIMD/parallel kernels where available and fall back to Candle
//! for unsupported devices, dtypes, or layouts. CUDA builds with NVCC available
//! additionally load Phase 14 PTX kernels for Flash Attention and fused layer
//! primitives.

pub mod cpu;
pub mod dispatch;
pub mod flash_attn;
pub mod fused_ffn;
pub mod fused_norm;
pub mod fused_rope;

pub use dispatch::KernelPath;
pub use dispatch::attention_forward;
pub use dispatch::attention_forward_candle;
pub use dispatch::attention_forward_train;
pub use dispatch::attention_path;
pub use dispatch::rms_norm;
pub use dispatch::rms_norm_path;
