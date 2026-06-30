use aarambh_ai_kernel::cpu::parallel_attn::{cpu_parallel_attn, cpu_sequential_attn};
use aarambh_ai_kernel::dispatch::{AttentionMaskKind, KernelPath, attention_forward_candle};
use aarambh_ai_kernel::{attention_forward, attention_path, rms_norm, rms_norm_path};
use candle_core::{DType, Device, Tensor};

fn flat_f32(tensor: &Tensor) -> Vec<f32> {
    tensor
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

fn max_abs_diff(lhs: &Tensor, rhs: &Tensor) -> f32 {
    flat_f32(lhs)
        .into_iter()
        .zip(flat_f32(rhs))
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f32::max)
}

fn patterned_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 31) as f32 - 15.0;
            centered * scale
        })
        .collect()
}

#[test]
fn rms_norm_dispatch_uses_cpu_simd_for_cpu_f32() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(patterned_values(2 * 3 * 65, 0.03), (2, 3, 65), &device).unwrap();
    let weight = Tensor::from_vec(patterned_values(65, 0.01), 65, &device).unwrap() + 1.0;
    let weight = weight.unwrap();

    assert_eq!(rms_norm_path(&x, &weight), KernelPath::CpuSimd);
}

#[test]
fn rms_norm_simd_matches_candle_reference_with_tail_lanes() {
    let device = Device::Cpu;
    let x = Tensor::from_vec(patterned_values(4 * 7 * 65, 0.02), (4, 7, 65), &device).unwrap();
    let weight = Tensor::from_vec(
        (0..65).map(|idx| 0.75 + idx as f32 * 0.001).collect(),
        65,
        &device,
    )
    .unwrap();

    let expected = candle_nn::ops::rms_norm(&x, &weight, 1e-5).unwrap();
    let actual = rms_norm(&x, &weight, 1e-5).unwrap();

    assert_eq!(actual.shape(), expected.shape());
    assert!(
        max_abs_diff(&actual, &expected) < 1e-4,
        "max diff was {}",
        max_abs_diff(&actual, &expected)
    );
}

#[test]
fn attention_dispatch_uses_cpu_parallel_for_cpu_f32() {
    let device = Device::Cpu;
    let q = Tensor::zeros((1, 2, 3, 4), DType::F32, &device).unwrap();
    let k = Tensor::zeros((1, 2, 5, 4), DType::F32, &device).unwrap();
    let v = Tensor::zeros((1, 2, 5, 4), DType::F32, &device).unwrap();

    assert_eq!(attention_path(&q, &k, &v, None), KernelPath::CpuParallel);
}

#[test]
fn attention_mask_kind_recognizes_project_causal_shape() {
    let device = Device::Cpu;
    let q = Tensor::zeros((1, 2, 3, 4), DType::F32, &device).unwrap();
    let k = Tensor::zeros((1, 2, 5, 4), DType::F32, &device).unwrap();
    let mask = Tensor::zeros((1, 1, 3, 5), DType::F32, &device).unwrap();

    assert_eq!(
        aarambh_ai_kernel::dispatch::attention_mask_kind(&q, &k, Some(&mask)),
        AttentionMaskKind::Causal
    );
}

#[test]
fn parallel_attention_matches_candle_without_mask() {
    let device = Device::Cpu;
    let q = Tensor::from_vec(
        patterned_values(2 * 3 * 5 * 16, 0.015),
        (2, 3, 5, 16),
        &device,
    )
    .unwrap();
    let k = Tensor::from_vec(
        patterned_values(2 * 3 * 7 * 16, 0.012),
        (2, 3, 7, 16),
        &device,
    )
    .unwrap();
    let v = Tensor::from_vec(
        patterned_values(2 * 3 * 7 * 16, 0.01),
        (2, 3, 7, 16),
        &device,
    )
    .unwrap();
    let scale = 1.0 / 16f64.sqrt();

    let expected = attention_forward_candle(&q, &k, &v, None, scale).unwrap();
    let actual = attention_forward(&q, &k, &v, None, scale).unwrap();
    let sequential = cpu_sequential_attn(&q, &k, &v, None, scale).unwrap();

    assert!(max_abs_diff(&actual, &expected) < 1e-4);
    assert!(max_abs_diff(&actual, &sequential) < 1e-6);
}

#[test]
fn parallel_attention_matches_candle_with_broadcast_mask() {
    let device = Device::Cpu;
    let q_seq = 5;
    let kv_seq = 7;
    let q = Tensor::from_vec(
        patterned_values(2 * 3 * q_seq * 16, 0.017),
        (2, 3, q_seq, 16),
        &device,
    )
    .unwrap();
    let k = Tensor::from_vec(
        patterned_values(2 * 3 * kv_seq * 16, 0.011),
        (2, 3, kv_seq, 16),
        &device,
    )
    .unwrap();
    let v = Tensor::from_vec(
        patterned_values(2 * 3 * kv_seq * 16, 0.013),
        (2, 3, kv_seq, 16),
        &device,
    )
    .unwrap();
    let mask_data = (0..q_seq)
        .flat_map(|q_idx| {
            (0..kv_seq).map(move |kv_idx| if kv_idx <= q_idx + 2 { 0.0 } else { -1.0e9 })
        })
        .collect::<Vec<f32>>();
    let mask = Tensor::from_vec(mask_data, (1, 1, q_seq, kv_seq), &device).unwrap();
    let scale = 1.0 / 16f64.sqrt();

    let expected = attention_forward_candle(&q, &k, &v, Some(&mask), scale).unwrap();
    let actual = cpu_parallel_attn(&q, &k, &v, Some(&mask), scale).unwrap();

    assert!(
        max_abs_diff(&actual, &expected) < 1e-4,
        "max diff was {}",
        max_abs_diff(&actual, &expected)
    );
}

#[test]
fn cuda_phase14_wrappers_report_kernel_availability() {
    assert_eq!(
        aarambh_ai_kernel::flash_attn::cuda_kernels_compiled(),
        cfg!(aarambh_cuda_kernels)
    );
    assert_eq!(
        aarambh_ai_kernel::fused_norm::cuda_kernels_compiled(),
        cfg!(aarambh_cuda_kernels)
    );
    assert_eq!(
        aarambh_ai_kernel::fused_rope::cuda_kernels_compiled(),
        cfg!(aarambh_cuda_kernels)
    );
    assert_eq!(
        aarambh_ai_kernel::fused_ffn::cuda_kernels_compiled(),
        cfg!(aarambh_cuda_kernels)
    );
}

#[cfg(all(feature = "cuda", aarambh_cuda_kernels))]
mod cuda_tests {
    use super::*;

    #[test]
    fn cuda_flash_attention_matches_candle_without_mask() {
        let device = Device::new_cuda(0).unwrap();
        let q = Tensor::from_vec(
            patterned_values(1 * 2 * 4 * 32, 0.011),
            (1, 2, 4, 32),
            &device,
        )
        .unwrap();
        let k = Tensor::from_vec(
            patterned_values(1 * 2 * 4 * 32, 0.013),
            (1, 2, 4, 32),
            &device,
        )
        .unwrap();
        let v = Tensor::from_vec(
            patterned_values(1 * 2 * 4 * 32, 0.017),
            (1, 2, 4, 32),
            &device,
        )
        .unwrap();
        let scale = 1.0 / 32f64.sqrt();

        assert_eq!(
            attention_path(&q, &k, &v, None),
            KernelPath::CudaFlashAttention
        );
        let expected = attention_forward_candle(&q, &k, &v, None, scale).unwrap();
        let actual = attention_forward(&q, &k, &v, None, scale).unwrap();
        assert!(
            max_abs_diff(
                &actual.to_device(&Device::Cpu).unwrap(),
                &expected.to_device(&Device::Cpu).unwrap()
            ) < 1e-4
        );
    }

    #[test]
    fn cuda_fused_rms_norm_matches_candle() {
        let device = Device::new_cuda(0).unwrap();
        let hidden = 64;
        let x = Tensor::from_vec(
            patterned_values(2 * 3 * hidden, 0.017),
            (2, 3, hidden),
            &device,
        )
        .unwrap();
        let weight = Tensor::from_vec(vec![1.0f32; hidden], hidden, &device).unwrap();
        assert_eq!(rms_norm_path(&x, &weight), KernelPath::CudaFusedRmsNorm);
        let expected = candle_nn::ops::rms_norm(&x, &weight, 1e-5).unwrap();
        let actual = rms_norm(&x, &weight, 1e-5).unwrap();
        assert!(
            max_abs_diff(
                &actual.to_device(&Device::Cpu).unwrap(),
                &expected.to_device(&Device::Cpu).unwrap()
            ) < 1e-4
        );
    }
}
