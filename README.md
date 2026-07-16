# aarambh-ai

> Sanskrit: *beginning*. A ground-up language model system in Rust.

[![CI](https://github.com/AarambhDevHub/aarambh-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AarambhDevHub/aarambh-ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.89%2B-orange.svg)](https://www.rust-lang.org)

aarambh-ai is a decoder-only language model implementation built with Rust and
Candle. The repository covers the full engineering path: tokenization, model
construction, training, inference, quantization, adapter tuning, alignment,
evaluation, multimodal input, safety, and an OpenAI-compatible server.

The production source release is **v2.0.0**. Current mainline development is
**v3.0.0-alpha.3**, with hybrid Gated DeltaNet, DeepSeek Sparse Attention, and
fine-grained MoE with shared experts.

> [!IMPORTANT]
> This is a source and engineering project. It does not publish crates to
> crates.io and does not ship pretrained checkpoints, adapters, GGUF files, or
> compiled binaries. You must train a model or provide compatible weights.

## What Is Included

| Area | Capabilities |
|---|---|
| Model | RMSNorm, RoPE, GQA, SwiGLU, KV cache, tied embeddings, Tiny to Large configs |
| Efficient architecture | YaRN/NTK/linear RoPE scaling, Gated DeltaNet, learned block-sparse DSA, coarse and fine-grained MoE |
| Training | BPE data pipeline, AdamW, cosine schedule, gradient accumulation/clipping, checkpoint resume, BF16 CUDA, single-node multi-GPU |
| Fine-tuning | SFT, LoRA, QLoRA, DoRA, QDoRA, VLM adapters, GRPO, DPO, QDPO, tool-call tuning |
| Inference | Greedy and sampled decoding, streaming, thinking budgets, speculative decoding, grammar-constrained tool calls |
| Model formats | SafeTensors, INT8, GPTQ/AWQ INT4, GGUF, Hugging Face conversion, quantized KV cache |
| Evaluation | Perplexity, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite, preference, recall, vision, and tool scorecards |
| Vision | Frozen CLIP-style encoder, projector pretraining, image fusion, VQA instruction tuning |
| Runtime | CPU SIMD, Rayon attention, optional custom CUDA PTX kernels, Axum 0.8.9 HTTP/SSE server |
| Guardrails | Prompt-injection checks, jailbreak checks, PII redaction, output scanning, streaming token safety, audit logs |
| Self-learning | Opt-in critique, replay buffer, verifier rewards, deferred CPU updates, CUDA vision mode |

The implementation history and proof obligations for each feature live in the
[roadmaps](#documentation). This README focuses on building and using the
project.

## Requirements

- Rust 1.89 or newer
- Linux or another platform supported by Candle
- Optional NVIDIA GPU and CUDA toolkit for `--features cuda`
- `nvcc` available at build time for custom CUDA PTX kernels
- Python 3 only for dataset preparation scripts

CPU builds do not require CUDA. Tiny smoke configurations are designed for
local development; Medium and Large training require suitable GPU memory.

## Quick Start

```sh
git clone https://github.com/AarambhDevHub/aarambh-ai.git
cd aarambh-ai

cargo check --workspace --all-targets --locked
cargo test --workspace --locked
cargo build --release --locked -p aarambh-ai
target/release/aarambh-ai --help
```

Run a two-step CPU training smoke test using the checked-in Tiny Shakespeare
fixture:

```sh
target/release/aarambh-ai train \
  --config configs/tiny_shakespeare_smoke.toml
```

Train the normal Tiny recipe:

```sh
target/release/aarambh-ai train \
  --config configs/tiny_shakespeare.toml
```

Training creates a tokenizer, model checkpoints, optimizer state, and
`latest.json`/`best.json` pointers under the configured checkpoint directory.
Smoke checkpoints validate the pipeline; two optimizer steps are not enough to
produce useful language quality.

## CLI

```text
aarambh-ai train       Pretrain or continue a configured model
aarambh-ai infer       Generate text or answer an image-grounded prompt
aarambh-ai eval        Run evaluation tasks and compare scorecards
aarambh-ai quantise    Calibrate and export INT8/INT4 GGUF checkpoints
aarambh-ai convert     Convert SafeTensors, GGUF, or Hugging Face layouts
aarambh-ai finetune    Run SFT, adapters, GRPO, DPO, VLM, or merge workflows
aarambh-ai selflearn   Manage replay and persistent self-learning state
aarambh-ai serve       Start the OpenAI-compatible HTTP/SSE server
```

Use `aarambh-ai <command> --help` for the complete option set.

## Common Workflows

### Train With CUDA

Build with CUDA explicitly:

```sh
cargo build --release --locked -p aarambh-ai --features cuda
scripts/phase13_prepare_wikitext103.sh data

target/release/aarambh-ai train \
  --config configs/wikitext103_cuda_smoke.toml
```

Representative training recipes:

| Config | Purpose |
|---|---|
| `configs/tiny_shakespeare_smoke.toml` | Fast CPU pipeline check |
| `configs/wikitext103_tiny.toml` | Tiny WikiText-103 training |
| `configs/wikitext103_small.toml` | Small CUDA training |
| `configs/wikitext103_medium.toml` | Medium CUDA training |
| `configs/wikitext103_large.toml` | Large CUDA training |
| `configs/wikitext103_small_2gpu.toml` | Single-node two-GPU data parallel training |
| `configs/medium_16k.toml` | Medium YaRN long-context continuation |
| `configs/medium_hybrid_dsa.toml` | Medium hybrid DSA continuation |
| `configs/medium_finegrained_moe.toml` | Medium fine-grained MoE retrofit |

For two GPUs, launch one process per rank with matching run IDs:

```sh
export AARAMBH_WORLD_SIZE=2
export AARAMBH_DIST_RUN_ID="wikitext-2gpu-$(date +%s)"

AARAMBH_RANK=0 AARAMBH_LOCAL_RANK=0 target/release/aarambh-ai train \
  --config configs/wikitext103_small_2gpu.toml &
AARAMBH_RANK=1 AARAMBH_LOCAL_RANK=1 target/release/aarambh-ai train \
  --config configs/wikitext103_small_2gpu.toml &
wait
```

Multi-GPU support is single-node data parallelism, not tensor parallelism or
multi-node training.

### Generate Text

```sh
target/release/aarambh-ai infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "To be, or not to be" \
  --max-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --stream
```

Useful inference options include:

- `--greedy` for deterministic argmax decoding
- `--thinking low|medium|high` for bounded reasoning-token budgets
- `--predict-view` for next-token probability inspection
- `--stats` for throughput, cache, sparse-attention, and MoE diagnostics
- `--safety strict|permissive|research|none` for policy selection
- `--image <path>` for vision-language inference
- `--speculative` plus draft-model options for exact speculative decoding
- `--tools <schema.json>` for grammar-constrained function calls

Tool calls are emitted and validated but never executed by aarambh-ai.

### Serve An OpenAI-Compatible API

```sh
target/release/aarambh-ai serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --model-id aarambh-tiny \
  --port 8080
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aarambh-tiny",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 32,
    "stream": true
  }'
```

Endpoints include `/v1/chat/completions`, `/v1/completions`, `/v1/models`,
`/healthz`, `/readyz`, and `/metrics`. External binds require
`AARAMBH_AI_API_KEY`. See [the server guide](docs/inference-server.md) for SDK,
safety, authentication, and batching details.

### Evaluate A Checkpoint

```sh
scripts/phase17_prepare_eval_sets.sh data/eval 128

target/release/aarambh-ai eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks ppl,mmlu,hellaswag,gsm8k \
  --data-dir data/eval \
  --out scorecard.json \
  --markdown scorecard.md
```

Compare two existing scorecards:

```sh
target/release/aarambh-ai eval \
  --compare before.json after.json \
  --markdown comparison.md
```

Generated-code execution is disabled unless `--allow-code-exec` is passed.

### Quantize And Convert

Create a calibrated Q4 GGUF checkpoint:

```sh
target/release/aarambh-ai quantise \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --method gptq \
  --bits 4 \
  --calibration-data data/tiny_shakespeare.txt \
  --samples 128 \
  --output checkpoints/tiny-q4.gguf
```

The same `infer --model` option accepts Aarambh SafeTensors and GGUF files.
Use `aarambh-ai convert --help` for Hugging Face conversion and direct format
conversion options.

### Fine-Tune And Merge

SFT input is JSONL with `instruction`, optional `thinking`, and `response`
fields. A minimal fixture is included at `data/instruct_tiny.jsonl`.

```sh
target/release/aarambh-ai finetune dora \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --output adapters/tiny_dora \
  --lora-rank 8

target/release/aarambh-ai finetune merge \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/best/model.safetensors \
  --adapter adapters/tiny_dora \
  --method auto \
  --output checkpoints/tiny_dora_merged
```

Available tuning commands are `sft`, `qlora`, `dora`, `qdora`, `vlm-dora`,
`vlm-qdora`, `grpo`, `dpo`, `qdpo`, `tool-sft`, and `tool-qlora`. Adapter
training for MoE base models is currently unsupported; train MoE models through
the pretraining path.

### Vision Smoke Test

The local fixture creates four synthetic images, a tiny vision encoder,
projector weights, tokenizer data, and VQA records:

```sh
python3 scripts/phase20_make_vqa_smoke_fixture.py

target/release/aarambh-ai finetune vlm-dora \
  --config configs/vision_vqa_smoke.toml \
  --base checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/vision_projector_smoke/tokenizer.json \
  --data data/vision_smoke/vqa_smoke_4.jsonl \
  --projector data/vision_smoke/projector_init.safetensors \
  --output adapters/vision_vqa_smoke \
  --lora-rank 4 \
  --max-steps 2
```

This proves the multimodal training path; four examples cannot produce a useful
vision-language model.

### Self-Learning

Self-learning is explicit and disabled by default. CPU mode records critique,
replay, and deferred adapter state; CUDA mode can apply supported online
updates. It is not autonomous production retraining.

```sh
target/release/aarambh-ai infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "Explain recursion simply." \
  --max-tokens 32 \
  --self-learn cpu \
  --replay-path data/replay.jsonl \
  --self-learn-state-dir adapters/selflearn

target/release/aarambh-ai selflearn stats \
  --replay-path data/replay.jsonl \
  --self-learn-state-dir adapters/selflearn
```

See the self-learning design documents for replay acceptance, verifier,
gradient-flush, vision-cache, and safety semantics.

## v3 Architecture Smokes

The current alpha line can be tested without a GPU:

```sh
# Gated DeltaNet recurrent state and backward path
target/release/aarambh-ai train --config configs/gated_deltanet_smoke.toml

# Learned block-sparse DSA teacher and sparse-only steps
target/release/aarambh-ai train --config configs/dsa_smoke.toml

# DSA + DeltaNet + fine-grained routed MoE + shared expert
target/release/aarambh-ai train --config configs/moe_finegrained_smoke.toml
```

Checkpoint retrofit and comparison tooling:

- `scripts/phase29_prepare_hybrid_retrofit.sh`
- `scripts/phase29_benchmark_hybrid.sh`
- `scripts/phase30_prepare_dsa_retrofit.sh`
- `scripts/phase30_benchmark_dsa.sh`
- `scripts/phase31_sweep_moe.sh`

The Phase 31 method and result contract are documented in
[docs/phase31_moe_sweep.md](docs/phase31_moe_sweep.md). Hardware benchmark
results are not claimed until the scripts have produced scorecards.

## Model Scales

| Scale | Parameters | Hidden | Layers | Heads | KV heads | FFN | Base context | RoPE theta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 25M | 384 | 8 | 6 | 2 | 1,024 | 512 | 10,000 |
| Small | 117M | 768 | 12 | 12 | 4 | 2,688 | 1,024 | 10,000 |
| Medium | 360M | 1,024 | 24 | 16 | 8 | 3,392 | 2,048 | 500,000 |
| Large | 1.3B | 2,048 | 24 | 32 | 8 | 6,656 | 4,096 | 500,000 |

All standard scales use a 32,000-token vocabulary, RMSNorm epsilon `1e-5`,
GQA, SwiGLU, and tied embeddings. Long-context and hybrid variants are selected
through TOML without changing the base scale definitions.

## Workspace

The workspace contains 16 internal library crates and one CLI package:

```text
aarambh-ai-core        Shared config, device, dtype, errors, and traits
aarambh-ai-tokenizer   BPE tokenizer and reserved special tokens
aarambh-ai-data        Datasets, preprocessing, sharding, and loaders
aarambh-ai-kernel      CPU SIMD and optional CUDA kernels
aarambh-ai-nn          Neural layers, attention, DeltaNet, DSA, and MoE
aarambh-ai-model       Full decoder model and cache integration
aarambh-ai-weights     SafeTensors, GGUF, conversion, and retrofit loading
aarambh-ai-quant       INT8/INT4, GPTQ, AWQ, QAT, and KV quantization
aarambh-ai-train       Optimizer, schedules, checkpointing, and distributed train
aarambh-ai-finetune    Adapters, SFT, GRPO, DPO, VLM, and tool tuning
aarambh-ai-inference   Sampling, caching, thinking, speculative, and tool grammar
aarambh-ai-safety      Input, output, streaming, PII, and audit policies
aarambh-ai-selflearn   Critique, replay, verifiers, and persistent update state
aarambh-ai-eval        Evaluation tasks, scorecards, and comparisons
aarambh-ai-vision      Image preprocessing, encoder, projector, and fusion
aarambh-ai-serve       Axum HTTP/SSE serving and continuous batching
aarambh-ai             Command-line application
```

Packages inherit one workspace version and use `publish = false`.

## Development Checks

Run the same primary gates used by CI:

```sh
cargo fmt --all --check
cargo check --workspace --all-targets --locked
cargo test --workspace --no-fail-fast --locked
cargo clippy --workspace --all-targets --locked -- \
  -D warnings -D clippy::undocumented_unsafe_blocks
RUSTDOCFLAGS="-D warnings -D missing_docs" \
  cargo doc --workspace --no-deps --locked
scripts/phase28_release_audit.sh
```

CUDA checks require a CUDA-capable environment and are intentionally opt-in.

## Documentation

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | v1 model, training, inference, safety, and self-learning design |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | v2 long context, vision, MoE, distributed, tools, and serving additions |
| [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) | v3 hybrid attention, DSA, fine-grained MoE, and planned architecture |
| [ROADMAP.md](ROADMAP.md) | Completed v1 phases |
| [ROADMAP_V2.md](ROADMAP_V2.md) | Completed v2 phases through the v2.0.0 release |
| [ROADMAP_V3.md](ROADMAP_V3.md) | Current v3 delivery plan and status |
| [SELF_LEARNING.md](SELF_LEARNING.md) | Text self-learning design |
| [SELF_LEARNING_V2.md](SELF_LEARNING_V2.md) | Vision-aware self-learning design |
| [SELF_LEARNING_V3.md](SELF_LEARNING_V3.md) | Planned v3 memory and forgetting diagnostics |
| [docs/aarambh-ai-config-toml-guide.md](docs/aarambh-ai-config-toml-guide.md) | Configuration field reference |
| [docs/aarambh-ai-complete-guide.md](docs/aarambh-ai-complete-guide.md) | Beginner-oriented project walkthrough |
| [docs/aarambh-ai-math-formulas-guide.md](docs/aarambh-ai-math-formulas-guide.md) | Mathematical foundations and worked examples |
| [docs/inference-server.md](docs/inference-server.md) | Server endpoints, SDK usage, auth, safety, and limits |
| [RELEASE.md](RELEASE.md) | Source-release process and artifact policy |
| [CHANGELOG.md](CHANGELOG.md) | Versioned implementation history |

## Current Boundaries

- No pretrained model is included, so output quality depends entirely on your
  training data, update count, and tuning.
- GGUF tensors are dequantized when loaded by the current universal model path;
  smaller files do not yet imply fully quantized compute.
- MoE dispatch computes every routed expert and applies dense weights. Fine
  granularity changes capacity and routing but is not sparse grouped dispatch.
- Multi-GPU support is single-node data parallel training.
- Tool calls are generated but never executed.
- The server currently hosts one text model and one generated choice per
  request; vision and self-learning are CLI workflows.
- HumanEval-style code execution requires explicit opt-in.

Additional exclusions and future work are tracked in the versioned roadmaps,
not hidden in this README.

## Contributing And Security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Use
[GitHub issues](https://github.com/AarambhDevHub/aarambh-ai/issues) for
reproducible bugs and scoped feature requests. Report vulnerabilities through
[SECURITY.md](SECURITY.md), not a public issue.

## Citation

```bibtex
@software{aarambh_ai_2026,
  title   = {aarambh-ai: A Ground-Up Language Model System in Rust},
  author  = {Aarambh Dev Hub},
  year    = {2026},
  url     = {https://github.com/AarambhDevHub/aarambh-ai},
  version = {2.0.0},
  license = {Apache-2.0}
}
```

## License

Licensed under the [Apache License 2.0](LICENSE).
