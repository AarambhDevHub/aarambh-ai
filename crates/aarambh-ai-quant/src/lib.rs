//! Quantisation, dequantisation, calibration, and GGUF-compatible packing utilities.
#![deny(missing_docs)]

/// Absolute-maximum int8 quantisation.
pub mod absmax;
/// Activation-aware weight quantisation.
pub mod awq;
/// Calibration statistics collection.
pub mod calibrate;
/// Dequantisation helpers.
pub mod dequant;
/// GGUF Q4_K_M block packing helpers.
pub mod gguf_quant;
/// GPTQ calibration and quantisation.
pub mod gptq;
/// Affine int4 packing and unpacking.
pub mod int4;
/// Quantised key/value cache utilities.
pub mod kv_quant;
/// Fake quantisation nodes for quantisation-aware training.
pub mod qat;
/// Shared quantisation data structures.
pub mod types;

pub use absmax::quantise_absmax_i8;
pub use awq::{compute_activation_scales, quantise_layer_awq};
pub use calibrate::{CalibrationStats, run_calibration};
pub use dequant::{dequantise_i4, dequantise_i8};
pub use gguf_quant::{
    Q4_K_M_BLOCK_SIZE, Q4_K_M_ENCODED_SIZE, dequantise_block_q4_k_m, f16_to_f32, f32_to_f16,
    quantise_block_q4_k_m,
};
pub use gptq::{cholesky_invert, compute_hessian, quantise_layer_gptq};
pub use int4::{dequantise_packed_i4_to_vec, pack_i4_values, quantise_affine_i4};
pub use kv_quant::QuantisedKvCache;
pub use qat::{FakeQuantNode, fake_quantise};
pub use types::{GgufFormat, I8QuantizedTensor, PackedInt4Tensor, QuantMethod, QuantizedTensor};
