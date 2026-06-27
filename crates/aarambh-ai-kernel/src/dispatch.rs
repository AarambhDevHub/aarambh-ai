use candle_core::{DType, Result, Tensor};

use crate::cpu::parallel_attn::cpu_parallel_attn;
use crate::cpu::simd_norm::cpu_rms_norm_simd;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPath {
    CpuSimd,
    CpuParallel,
    CandleFallback,
}

pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    if rms_norm_path(x, weight) == KernelPath::CpuSimd
        && let Ok(out) = cpu_rms_norm_simd(x, weight, eps)
    {
        return Ok(out);
    }
    candle_nn::ops::rms_norm(x, weight, eps)
}

pub fn rms_norm_path(x: &Tensor, weight: &Tensor) -> KernelPath {
    if x.device().same_device(weight.device())
        && x.device().is_cpu()
        && x.dtype() == DType::F32
        && weight.dtype() == DType::F32
        && x.dims().last().copied() == weight.dims().first().copied()
        && weight.dims().len() == 1
    {
        KernelPath::CpuSimd
    } else {
        KernelPath::CandleFallback
    }
}

pub fn attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
) -> Result<Tensor> {
    if attention_path(q, k, v, mask) == KernelPath::CpuParallel
        && let Ok(out) = cpu_parallel_attn(q, k, v, mask, scale)
    {
        return Ok(out);
    }
    attention_forward_candle(q, k, v, mask, scale)
}

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
    } else {
        KernelPath::CandleFallback
    }
}

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
