use candle_core::{DType, Result, Tensor};

use crate::cpu::parallel_attn::cpu_parallel_attn;
use crate::cpu::simd_norm::cpu_rms_norm_simd;
use crate::flash_attn::{flash_attention_forward, flash_attention_forward_train};
use crate::fused_norm::fused_rms_norm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Kernel implementation chosen for a supported operation.
pub enum KernelPath {
    /// CPU SIMD kernel.
    CpuSimd,
    /// CPU parallel kernel.
    CpuParallel,
    /// CUDA Flash Attention kernel.
    CudaFlashAttention,
    /// CUDA fused RMSNorm kernel.
    CudaFusedRmsNorm,
    /// Candle fallback path.
    CandleFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention mask representation understood by kernel dispatch.
pub enum AttentionMaskKind {
    /// No mask was supplied.
    None,
    /// Standard causal mask.
    Causal,
    /// Additive attention bias mask.
    Additive,
}

/// Run RMSNorm with the fastest available kernel and fall back to Candle.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    match rms_norm_path(x, weight) {
        KernelPath::CpuSimd => {
            if let Ok(out) = cpu_rms_norm_simd(x, weight, eps) {
                return Ok(out);
            }
        }
        KernelPath::CudaFusedRmsNorm => {
            if let Ok(out) = fused_rms_norm(x, weight, eps) {
                return Ok(out);
            }
        }
        _ => {}
    }
    candle_nn::ops::rms_norm(x, weight, eps)
}

/// Return the RMSNorm kernel path that would be selected for the tensors.
pub fn rms_norm_path(x: &Tensor, weight: &Tensor) -> KernelPath {
    if x.device().same_device(weight.device())
        && x.device().is_cpu()
        && x.dtype() == DType::F32
        && weight.dtype() == DType::F32
        && x.dims().last().copied() == weight.dims().first().copied()
        && weight.dims().len() == 1
    {
        KernelPath::CpuSimd
    } else if cfg!(all(feature = "cuda", aarambh_cuda_kernels))
        && x.device().same_device(weight.device())
        && x.device().is_cuda()
        && matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && x.dtype() == weight.dtype()
        && x.dims().last().copied() == weight.dims().first().copied()
        && weight.dims().len() == 1
    {
        KernelPath::CudaFusedRmsNorm
    } else {
        KernelPath::CandleFallback
    }
}

/// Run inference attention with the fastest available kernel and fall back to Candle.
pub fn attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    match attention_path(q, k, v, mask) {
        KernelPath::CpuParallel => {
            if let Ok(out) = cpu_parallel_attn(q, k, v, mask, scale) {
                return Ok(out);
            }
        }
        KernelPath::CudaFlashAttention => {
            let causal = attention_mask_kind(q, k, mask) == AttentionMaskKind::Causal;
            if let Ok(out) = flash_attention_forward(q, k, v, causal, scale) {
                return Ok(out);
            }
        }
        _ => {}
    }
    attention_forward_candle(q, k, v, mask, scale)
}

/// Run training attention with the fastest available kernel and fall back to Candle.
pub fn attention_forward_train(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    if attention_path(q, k, v, mask) == KernelPath::CudaFlashAttention {
        let causal = attention_mask_kind(q, k, mask) == AttentionMaskKind::Causal;
        if let Ok(out) = flash_attention_forward_train(q, k, v, causal, scale) {
            return Ok(out);
        }
    }
    attention_forward_candle(q, k, v, mask, scale)
}

/// Return the attention kernel path that would be selected for the tensors.
pub fn attention_path(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> KernelPath {
    let Some((&qb, &qh, &_qq, &qd)) = dims4(q).as_ref().map(|d| (&d[0], &d[1], &d[2], &d[3]))
    else {
        return KernelPath::CandleFallback;
    };
    let Some((&kb, &kh, &_kk, &kd)) = dims4(k).as_ref().map(|d| (&d[0], &d[1], &d[2], &d[3]))
    else {
        return KernelPath::CandleFallback;
    };
    let Some((&vb, &vh, &_vk, &vd)) = dims4(v).as_ref().map(|d| (&d[0], &d[1], &d[2], &d[3]))
    else {
        return KernelPath::CandleFallback;
    };

    let mask_supported = mask.is_none_or(|m| m.device().is_cpu() && m.dtype() == DType::F32);
    if q.device().is_cpu()
        && q.device().same_device(k.device())
        && q.device().same_device(v.device())
        && q.dtype() == DType::F32
        && k.dtype() == DType::F32
        && v.dtype() == DType::F32
        && qb == kb
        && qb == vb
        && qh == kh
        && qh == vh
        && qd == kd
        && qd == vd
        && mask_supported
    {
        KernelPath::CpuParallel
    } else if cfg!(all(feature = "cuda", aarambh_cuda_kernels))
        && q.device().is_cuda()
        && q.device().same_device(k.device())
        && q.device().same_device(v.device())
        && matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
        && q.dtype() == k.dtype()
        && q.dtype() == v.dtype()
        && qb == kb
        && qb == vb
        && qh == kh
        && qh == vh
        && qd == kd
        && qd == vd
        && qd <= 256
        && matches!(
            attention_mask_kind(q, k, mask),
            AttentionMaskKind::None | AttentionMaskKind::Causal
        )
    {
        KernelPath::CudaFlashAttention
    } else {
        KernelPath::CandleFallback
    }
}

/// Classify an attention mask for kernel dispatch.
pub fn attention_mask_kind(q: &Tensor, k: &Tensor, mask: Option<&Tensor>) -> AttentionMaskKind {
    let Some(mask) = mask else {
        return AttentionMaskKind::None;
    };
    let Some([_, _, q_len, _]) = dims4(q) else {
        return AttentionMaskKind::Additive;
    };
    let Some([_, _, kv_len, _]) = dims4(k) else {
        return AttentionMaskKind::Additive;
    };
    if mask.dims() == [1, 1, q_len, kv_len] {
        AttentionMaskKind::Causal
    } else {
        AttentionMaskKind::Additive
    }
}

/// Candle implementation of scaled dot-product attention.
pub fn attention_forward_candle(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    let attn_weights = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
    let attn_weights = match mask {
        Some(mask) => attn_weights.broadcast_add(mask)?,
        None => attn_weights,
    };
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    attn_weights.matmul(v)
}

fn dims4(tensor: &Tensor) -> Option<[usize; 4]> {
    let dims = tensor.dims();
    (dims.len() == 4).then(|| [dims[0], dims[1], dims[2], dims[3]])
}
