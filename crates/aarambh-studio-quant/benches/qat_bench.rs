use aarambh_studio_core::{QatConfig, QatTarget, QuantBits, QuantGranularity};
use aarambh_studio_quant::{FakeQuantize, QatContext, QatLinear};
use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn qat_benchmarks(criterion: &mut Criterion) {
    let device = Device::Cpu;
    let weight = Tensor::randn(0.0f32, 0.02, (384, 384), &device).unwrap();
    let input = Tensor::randn(0.0f32, 1.0, (8, 64, 384), &device).unwrap();

    let mut fake_quant = criterion.benchmark_group("qat_fake_quant");
    for bits in [QuantBits::Int4, QuantBits::Int8] {
        let quantizer = FakeQuantize::new(bits, QuantGranularity::ExportAligned);
        fake_quant.bench_with_input(
            BenchmarkId::new("export_aligned", bits.bits()),
            &bits,
            |bencher, _| bencher.iter(|| quantizer.simulate(&weight).unwrap()),
        );
    }
    fake_quant.finish();

    let context = QatContext::new(QatConfig::default()).unwrap();
    let linear = QatLinear::new(
        Linear::new(weight, None),
        QatTarget::Attention,
        Some(context.clone()),
    );
    linear.forward(&input).unwrap();
    criterion.bench_function("qat_linear_cached_forward", |bencher| {
        bencher.iter(|| linear.forward(&input).unwrap())
    });
    criterion.bench_function("qat_linear_refresh_and_forward", |bencher| {
        bencher.iter(|| {
            context.advance_generation();
            linear.forward(&input).unwrap()
        })
    });

    let plain = Tensor::ones((1,), DType::F32, &device).unwrap();
    criterion.bench_function("qat_generation_advance", |bencher| {
        bencher.iter(|| {
            context.advance_generation();
            &plain
        })
    });
}

criterion_group!(benches, qat_benchmarks);
criterion_main!(benches);
