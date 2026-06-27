//! Kernel dispatch and optimized CPU kernels for aarambh-ai.
//!
//! Phase 4 keeps CUDA as build-time preparation only. Runtime dispatch uses
//! optimized CPU kernels for supported CPU F32 tensors and falls back to Candle
//! for unsupported devices, dtypes, or layouts.

pub mod cpu;
pub mod dispatch;
pub mod flash_attn;
pub mod fused_ffn;
pub mod fused_norm;
pub mod fused_rope;

pub use dispatch::KernelPath;
pub use dispatch::attention_forward;
pub use dispatch::attention_forward_candle;
pub use dispatch::attention_path;
pub use dispatch::rms_norm;
pub use dispatch::rms_norm_path;
