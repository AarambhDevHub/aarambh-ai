use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp2, Error, Layout, Result, Shape, Tensor};
use rayon::prelude::*;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// SIMD implementation selected for CPU RMSNorm.
pub enum SimdKind {
    /// AVX-512 implementation.
    Avx512,
    /// AVX2 implementation with FMA.
    Avx2Fma,
    /// AVX2 implementation without FMA-specific path.
    Avx2,
    /// Portable scalar fallback.
    Scalar,
}

/// Return the cached runtime SIMD selection.
pub fn selected_simd() -> SimdKind {
    static SELECTED: OnceLock<SimdKind> = OnceLock::new();
    *SELECTED.get_or_init(selected_simd_impl)
}

/// Run RMSNorm through the selected CPU SIMD custom op.
pub fn cpu_rms_norm_simd(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let dims = x.dims();
    let Some(&hidden) = dims.last() else {
        return Err(Error::msg("rms_norm expects rank >= 1"));
    };
    if hidden == 0 {
        return Err(Error::msg("rms_norm hidden dimension must be non-zero"));
    }
    if weight.dims() != [hidden] {
        return Err(Error::msg(format!(
            "rms_norm weight must have shape [{hidden}], got {:?}",
            weight.dims()
        )));
    }

    x.apply_op2_no_bwd(
        weight,
        &SimdRmsNorm {
            eps,
            simd: selected_simd(),
        },
    )
}

/// Portable scalar RMSNorm implementation over one row.
pub fn cpu_rms_norm_scalar(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let sum_sq = sum_squares_scalar(input);
    let inv_rms = (sum_sq / input.len() as f32 + eps).sqrt().recip();
    for ((dst, src), weight) in output.iter_mut().zip(input).zip(weight) {
        *dst = *src * inv_rms * *weight;
    }
}

#[derive(Debug, Clone, Copy)]
struct SimdRmsNorm {
    eps: f32,
    simd: SimdKind,
}

impl CustomOp2 for SimdRmsNorm {
    fn name(&self) -> &'static str {
        "aarambh-rms-norm-simd"
    }

    fn cpu_fwd(
        &self,
        input_storage: &CpuStorage,
        input_layout: &Layout,
        weight_storage: &CpuStorage,
        weight_layout: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let input = match input_storage {
            CpuStorage::F32(values) => contiguous_slice(values, input_layout, self.name())?,
            storage => {
                return Err(Error::UnsupportedDTypeForOp(storage.dtype(), self.name()).bt());
            }
        };
        let weight = match weight_storage {
            CpuStorage::F32(values) => contiguous_slice(values, weight_layout, self.name())?,
            storage => {
                return Err(Error::UnsupportedDTypeForOp(storage.dtype(), self.name()).bt());
            }
        };

        let dims = input_layout.shape().dims();
        let Some(&hidden) = dims.last() else {
            return Err(Error::msg("rms_norm expects rank >= 1"));
        };
        if hidden == 0 {
            return Err(Error::msg("rms_norm hidden dimension must be non-zero"));
        }
        if weight.len() != hidden {
            return Err(Error::msg(format!(
                "rms_norm weight must have {hidden} elements, got {}",
                weight.len()
            )));
        }
        if !input.len().is_multiple_of(hidden) {
            return Err(Error::msg(format!(
                "input element count {} is not divisible by hidden size {hidden}",
                input.len()
            )));
        }

        let mut output = vec![0f32; input.len()];
        input
            .par_chunks(hidden)
            .zip(output.par_chunks_mut(hidden))
            .for_each(|(src, dst)| rms_norm_row(src, weight, self.eps, dst, self.simd));

        Ok((CpuStorage::F32(output), Shape::from_dims(dims)))
    }
}

fn contiguous_slice<'a>(values: &'a [f32], layout: &Layout, op: &'static str) -> Result<&'a [f32]> {
    match layout.contiguous_offsets() {
        Some((start, end)) => Ok(&values[start..end]),
        None => Err(Error::RequiresContiguous { op }.bt()),
    }
}

fn rms_norm_row(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32], simd: SimdKind) {
    debug_assert_eq!(input.len(), weight.len());
    debug_assert_eq!(input.len(), output.len());

    match simd {
        SimdKind::Avx512 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                // SAFETY: selected_simd only returns Avx512 when the CPU supports avx512f.
                unsafe { rms_norm_row_avx512(input, weight, eps, output) };
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            cpu_rms_norm_scalar(input, weight, eps, output);
        }
        SimdKind::Avx2Fma => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                // SAFETY: selected_simd only returns Avx2Fma when the CPU supports avx2 and fma.
                unsafe { rms_norm_row_avx2_fma(input, weight, eps, output) };
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            cpu_rms_norm_scalar(input, weight, eps, output);
        }
        SimdKind::Avx2 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                // SAFETY: selected_simd only returns Avx2 when the CPU supports avx2.
                unsafe { rms_norm_row_avx2(input, weight, eps, output) };
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            cpu_rms_norm_scalar(input, weight, eps, output);
        }
        SimdKind::Scalar => cpu_rms_norm_scalar(input, weight, eps, output),
    }
}

fn sum_squares_scalar(input: &[f32]) -> f32 {
    input.iter().map(|value| value * value).sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn selected_simd_impl() -> SimdKind {
    if let Ok(force) = std::env::var("AARAMBH_SIMD_FORCE") {
        match force.to_ascii_lowercase().as_str() {
            "avx512" if std::arch::is_x86_feature_detected!("avx512f") => return SimdKind::Avx512,
            "avx2"
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma") =>
            {
                return SimdKind::Avx2Fma;
            }
            "avx2" if std::arch::is_x86_feature_detected!("avx2") => return SimdKind::Avx2,
            "scalar" => return SimdKind::Scalar,
            _ => {}
        }
    }

    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        SimdKind::Avx2Fma
    } else if std::arch::is_x86_feature_detected!("avx2") {
        SimdKind::Avx2
    } else if std::arch::is_x86_feature_detected!("avx512f") {
        SimdKind::Avx512
    } else {
        SimdKind::Scalar
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn selected_simd_impl() -> SimdKind {
    SimdKind::Scalar
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn rms_norm_row_avx512(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mut idx = 0;
    let mut acc = _mm512_setzero_ps();
    while idx + 16 <= input.len() {
        // SAFETY: idx advances in blocks of 16 and stays within input bounds.
        let values = unsafe { _mm512_loadu_ps(input.as_ptr().add(idx)) };
        acc = _mm512_fmadd_ps(values, values, acc);
        idx += 16;
    }

    let mut sum_sq = _mm512_reduce_add_ps(acc);
    for value in &input[idx..] {
        sum_sq += value * value;
    }

    let inv_rms = (sum_sq / input.len() as f32 + eps).sqrt().recip();
    let scale = _mm512_set1_ps(inv_rms);
    idx = 0;
    while idx + 16 <= input.len() {
        // SAFETY: idx advances in blocks of 16 and all three slices have equal length.
        let values = unsafe { _mm512_loadu_ps(input.as_ptr().add(idx)) };
        let weights = unsafe { _mm512_loadu_ps(weight.as_ptr().add(idx)) };
        let out = _mm512_mul_ps(_mm512_mul_ps(values, scale), weights);
        unsafe { _mm512_storeu_ps(output.as_mut_ptr().add(idx), out) };
        idx += 16;
    }
    for ((dst, src), weight) in output[idx..]
        .iter_mut()
        .zip(&input[idx..])
        .zip(&weight[idx..])
    {
        *dst = *src * inv_rms * *weight;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_row_avx2_fma(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mut idx = 0;
    let mut acc = _mm256_setzero_ps();
    while idx + 8 <= input.len() {
        // SAFETY: idx advances in blocks of 8 and stays within input bounds.
        let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(idx)) };
        acc = _mm256_fmadd_ps(values, values, acc);
        idx += 8;
    }

    let mut lanes = [0f32; 8];
    // SAFETY: lanes has exactly 8 f32 elements.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum_sq = lanes.iter().sum::<f32>();
    for value in &input[idx..] {
        sum_sq += value * value;
    }

    let inv_rms = (sum_sq / input.len() as f32 + eps).sqrt().recip();
    let scale = _mm256_set1_ps(inv_rms);
    idx = 0;
    while idx + 8 <= input.len() {
        // SAFETY: idx advances in blocks of 8 and all three slices have equal length.
        let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(idx)) };
        let weights = unsafe { _mm256_loadu_ps(weight.as_ptr().add(idx)) };
        let out = _mm256_mul_ps(_mm256_mul_ps(values, scale), weights);
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(idx), out) };
        idx += 8;
    }
    for ((dst, src), weight) in output[idx..]
        .iter_mut()
        .zip(&input[idx..])
        .zip(&weight[idx..])
    {
        *dst = *src * inv_rms * *weight;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn rms_norm_row_avx2(input: &[f32], weight: &[f32], eps: f32, output: &mut [f32]) {
    let mut idx = 0;
    let mut acc = _mm256_setzero_ps();
    while idx + 8 <= input.len() {
        // SAFETY: idx advances in blocks of 8 and stays within input bounds.
        let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(idx)) };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(values, values));
        idx += 8;
    }

    let mut lanes = [0f32; 8];
    // SAFETY: lanes has exactly 8 f32 elements.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), acc) };
    let mut sum_sq = lanes.iter().sum::<f32>();
    for value in &input[idx..] {
        sum_sq += value * value;
    }

    let inv_rms = (sum_sq / input.len() as f32 + eps).sqrt().recip();
    let scale = _mm256_set1_ps(inv_rms);
    idx = 0;
    while idx + 8 <= input.len() {
        // SAFETY: idx advances in blocks of 8 and all three slices have equal length.
        let values = unsafe { _mm256_loadu_ps(input.as_ptr().add(idx)) };
        let weights = unsafe { _mm256_loadu_ps(weight.as_ptr().add(idx)) };
        let out = _mm256_mul_ps(_mm256_mul_ps(values, scale), weights);
        unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(idx), out) };
        idx += 8;
    }
    for ((dst, src), weight) in output[idx..]
        .iter_mut()
        .zip(&input[idx..])
        .zip(&weight[idx..])
    {
        *dst = *src * inv_rms * *weight;
    }
}
