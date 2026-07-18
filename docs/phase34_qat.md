# Phase 34 Native QAT

Phase 34 adds weight-only INT4/INT8 quantization-aware training to the shared
pretraining loop. It does not add a separate adapter method and it does not
change the GGUF format.

## Execution Contract

- `AarambhModel::new_for_training` activates QAT when `[model.qat]` exists.
- `AarambhModel::new` always uses master weights directly, including inference,
  conversion, evaluation, and normal checkpoint loading.
- Covered linear projections see quantize/dequantize noise in their forward
  pass and an identity straight-through gradient in backward.
- Master weights and AdamW state remain floating point.
- Embeddings, RMSNorm, convolution kernels, and recurrent scalar parameters
  remain unwrapped.
- Fake-quantized weights are cached until the next optimizer generation, so
  gradient accumulation reuses one materialization.

## Quantization Policies

```toml
[model.qat]
bits = "int4" # int4 | int8
granularity = "export_aligned" # export_aligned | per_tensor | per_output_channel
targets = ["attention", "ffn", "moe_router", "delta_net", "dsa_indexer", "mtp"]
```

`export_aligned` is the production policy:

| Width | Simulation |
|---|---|
| INT4 | Q4_K_M blocks of 256 weights, zero-padded tail, f16 scale/min reconstruction |
| INT8 | One global absmax scale with signed `[-127, 127]` codes |
| DSA indexer | Forced INT8 under export-aligned policy |

The LM head is opt-in because tied embeddings can make its fake-quantized
matmul expensive and the embedding lookup itself remains floating point.

## Run The CPU Smoke

```sh
cargo build --release -p aarambh-ai
scripts/phase34_smoke.sh
```

The smoke runs two optimizer steps and checks that the model checkpoint and
QAT policy in `train_state.json` were written. Training logs include QAT bit
width, granularity, covered tensors/parameters, cache generation, and refresh
count.

## Continue A Tiny Checkpoint

`configs/qat_tiny.toml` expects the standard Tiny checkpoint and tokenizer:

```sh
cargo run --release -p aarambh-ai -- train --config configs/qat_tiny.toml
```

`retrofit_from` uses strict model-only loading for QAT. Missing, unexpected, or
shape-mismatched tensors fail. `resume = true` instead restores model weights,
optimizer moments, counters, and the persisted QAT policy; changing bits,
granularity, targets, or QAT enabled state is rejected.

## Export

SafeTensors checkpoints contain floating-point master weights. Export with the
existing path:

```sh
cargo run --release -p aarambh-ai -- convert \
  --config configs/qat_tiny.toml \
  --input checkpoints/qat_tiny/step_001000/model.safetensors \
  --output checkpoints/qat_tiny/model-q4.gguf \
  --gguf \
  --format q4_k_m
```

No QAT-specific file format exists.

## Validate Robustness

```sh
scripts/phase34_compare_qat.sh \
  configs/qat_tiny.toml \
  checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  checkpoints/qat_tiny/step_001000/model.safetensors \
  data/eval \
  reports/qat_tiny
```

The command evaluates four cases with identical tasks and limits:

1. Baseline floating-point checkpoint.
2. Baseline exported at the configured Q4/Q8 width.
3. QAT floating-point master checkpoint.
4. QAT checkpoint exported through the same Q4/Q8 path.

For each task, positive `baseline_quantization_drop` and
`qat_quantization_drop` mean degradation, regardless of metric direction.
`robustness_recovery = baseline_drop - qat_drop`; positive recovery means the
QAT checkpoint retained more quality after export. The repository does not
claim recovery until this report is produced from trained checkpoints.

## Benchmarks And Tests

```sh
cargo test -p aarambh-ai-quant qat::tests
cargo test -p aarambh-ai-model qat_is_enabled_only_for_training_construction
cargo test -p aarambh-ai-train qat_two_step_smoke_updates_weights_and_cache_generation
cargo bench -p aarambh-ai-quant --bench qat_bench
```

Unit tests cover exact Q4/Q8 exporter parity, padded Q4 tails, identity STE
gradients, cache reuse/invalidation, training-only activation, and two-step
optimizer updates. Hardware timing and checkpoint quality remain measured
outputs, not source-level assumptions.
