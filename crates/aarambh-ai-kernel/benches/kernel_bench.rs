use aarambh_ai_kernel::cpu::parallel_attn::{cpu_parallel_attn, cpu_sequential_attn};
use aarambh_ai_kernel::cpu::simd_norm::cpu_rms_norm_simd;
use candle_core::{Device, Tensor};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn patterned_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            let centered = (idx % 31) as f32 - 15.0;
            centered * scale
        })
        .collect()
}

fn bench_rms_norm(c: &mut Criterion) {
    let device = Device::Cpu;
    let hidden = 384;
    let x = Tensor::from_vec(
        patterned_values(4 * 512 * hidden, 0.01),
        (4, 512, hidden),
        &device,
    )
    .unwrap();
    let weight = Tensor::from_vec(vec![1.0f32; hidden], hidden, &device).unwrap();

    let mut group = c.benchmark_group("rms_norm");
    group.bench_function("candle", |b| {
        b.iter(|| {
            candle_nn::ops::rms_norm(black_box(&x), black_box(&weight), black_box(1e-5)).unwrap()
        })
    });
    group.bench_function("simd", |b| {
        b.iter(|| cpu_rms_norm_simd(black_box(&x), black_box(&weight), black_box(1e-5)).unwrap())
    });
    group.finish();
}

fn bench_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let batch = 2;
    let heads = 6;
    let q_seq = 64;
    let kv_seq = 64;
    let head_dim = 64;
    let q = Tensor::from_vec(
        patterned_values(batch * heads * q_seq * head_dim, 0.01),
        (batch, heads, q_seq, head_dim),
        &device,
    )
    .unwrap();
    let k = Tensor::from_vec(
        patterned_values(batch * heads * kv_seq * head_dim, 0.01),
        (batch, heads, kv_seq, head_dim),
        &device,
    )
    .unwrap();
    let v = Tensor::from_vec(
        patterned_values(batch * heads * kv_seq * head_dim, 0.01),
        (batch, heads, kv_seq, head_dim),
        &device,
    )
    .unwrap();
    let scale = 1.0 / (head_dim as f64).sqrt();

    let mut group = c.benchmark_group("attention");
    group.bench_function("sequential", |b| {
        b.iter(|| {
            cpu_sequential_attn(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                None,
                black_box(scale),
            )
            .unwrap()
        })
    });
    group.bench_function("parallel", |b| {
        b.iter(|| {
            cpu_parallel_attn(
                black_box(&q),
                black_box(&k),
                black_box(&v),
                None,
                black_box(scale),
            )
            .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, bench_rms_norm, bench_attention);
criterion_main!(benches);
