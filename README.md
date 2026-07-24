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
**v3.0.0-alpha.9**, with hybrid Gated DeltaNet, DeepSeek Sparse Attention,
fine-grained MoE with shared experts, Multi-Token Prediction (MTP), on-policy
distillation, native quantization-aware training, native video/document input,
and bounded long-horizon tool-use chains.

> [!IMPORTANT]
> This is a source and engineering project. It does not publish crates to
> crates.io and does not ship pretrained checkpoints, adapters, GGUF files, or
> compiled binaries. You must train a model or provide compatible weights.

## What Is Included

| Area | Capabilities |
|---|---|
| Model | RMSNorm, RoPE, GQA, SwiGLU, KV cache, tied embeddings, Tiny to Large configs |
| Efficient architecture | YaRN/NTK/linear RoPE scaling, Gated DeltaNet, learned block-sparse DSA, fine-grained MoE, MTP |
| Training | BPE data pipeline, AdamW, cosine schedule, gradient accumulation/clipping, checkpoint resume, BF16 CUDA, single-node multi-GPU, on-policy distillation, native INT4/INT8 QAT |
| Fine-tuning | SFT, LoRA, QLoRA, DoRA, QDoRA, VLM adapters, GRPO, DPO, QDPO, tool-call tuning |
| Inference | Greedy/sampled decoding, streaming, thinking budgets, external or one-checkpoint MTP speculation, tool grammar, caller-executed chains |
| Model formats | SafeTensors, INT8, GPTQ/AWQ INT4, GGUF, Hugging Face conversion, quantized KV cache |
| Evaluation | Perplexity, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite, preference, recall, vision, document ANLS, and tool scorecards |
| Vision | Frozen CLIP-style encoder, image/video/document fusion, temporal and 2D layout encoding, multimodal DoRA/QDoRA tuning |
| Runtime | CPU SIMD, Rayon attention, optional custom CUDA PTX kernels, Axum 0.8.9 HTTP/SSE server |
| Guardrails | Prompt-injection checks, jailbreak checks, PII redaction, output scanning, streaming token safety, audit logs |
| Self-learning | Opt-in critique, replay buffer, verifier rewards, deferred CPU updates, CUDA vision mode |

The implementation history and proof obligations for each feature live in the
[roadmaps](#documentation). This README focuses on building and using the
project.

## Requirements

- Rust 1.89 or newer
- Linux or another platform supported by Candle
- A C/C++ build toolchain for the bundled OpenH264 decoder
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
aarambh-ai infer       Generate text or answer an image/video-grounded prompt
aarambh-ai agent       Orchestrate bounded caller-executed tool-use chains
aarambh-ai eval        Run evaluation tasks and compare scorecards
aarambh-ai quantise    Calibrate and export INT8/INT4 GGUF checkpoints
aarambh-ai convert     Convert SafeTensors, GGUF, or Hugging Face layouts
aarambh-ai finetune    Run SFT, adapters, GRPO, DPO, VLM, or merge workflows
aarambh-ai distill     Train/evaluate on-policy or offline teacher distillation
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
| `configs/mtp_smoke.toml` | Two-step CPU MTP training check |
| `configs/medium_mtp.toml` | Medium Phase 31-to-MTP continuation |
| `configs/large_mtp.toml` | Large Phase 31-to-MTP continuation |
| `configs/distill_smoke.toml` | Phase 33 local CPU execution check |
| `configs/medium_distill.toml` | Medium on-policy distillation recipe |
| `configs/large_distill.toml` | Large on-policy distillation recipe |
| `configs/qat_smoke.toml` | Two-step CPU native-QAT execution check |
| `configs/qat_tiny.toml` | Tiny INT4 QAT continuation from an exact checkpoint |

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
- `--video <path>` with `--frames` and `--frame-sampling uniform|scene-aware`
  for visual-only H.264 MP4 inference
- `--document <pdf-or-image>` with optional comma-separated `--pages 1,3`
  for page-rendered document inference
- `--speculative` for one-checkpoint MTP, or add draft-model options for the
  external path
- `--tools <schema.json>` for grammar-constrained function calls

Tool calls are emitted and validated but never executed by aarambh-ai.

### Run A Tool-Use Chain

`agent` repeatedly decodes one schema-valid call, reads one caller-executed
result, and continues until the model emits a normal final response. Interactive
mode reads one `ToolResult` JSON object per line from stdin:

```sh
target/release/aarambh-ai agent \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tool_sft/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tools data/agent_tools_smoke.json \
  --prompt "Find the shipping price for customer C-17's latest order." \
  --max-steps 8 --greedy --safety strict
```

For deterministic replay/evaluation, pass a JSONL response path:

```sh
target/release/aarambh-ai agent \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tool_sft/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tools data/agent_tools_smoke.json \
  --results data/agent_results_smoke.jsonl \
  --prompt "Find the shipping price for customer C-17's latest order." \
  --max-steps 8 --jsonl
```

Results may contain bounded text/errors or an image, video, or document path
under `--result-root`. Native media is projected for the immediately following
decision and then retained as text metadata. `drop-oldest` is the default
context policy; `--eviction summarise` compacts evicted exchanges. Aarambh
never runs the tool, shell command, network request, or plugin itself.

### Understand Video

Phase 35 uses the existing frozen CLIP encoder for sampled frames, adds a
learned or sinusoidal temporal position, and fuses the frame blocks through
the same language-model input path used for images. Runtime decoding is native
H.264-in-MP4 through bundled OpenH264 and does not invoke FFmpeg.

Existing image-era tokenizers and SafeTensors checkpoints need a one-time,
function-preserving three-token vocabulary expansion:

```sh
target/release/aarambh-ai convert \
  --config configs/vision_vqa_smoke.toml \
  --input checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --output checkpoints/video_smoke/model.safetensors \
  --tokenizer checkpoints/vision_projector_smoke/tokenizer.json \
  --output-tokenizer checkpoints/video_smoke/tokenizer.json \
  --upgrade-video-vocab
```

After video instruction tuning, run inference with the saved projector and
temporal weights referenced by the config:

```sh
target/release/aarambh-ai infer \
  --config configs/video_qa_smoke_infer.toml \
  --model checkpoints/video_smoke/model.safetensors \
  --tokenizer checkpoints/video_smoke/tokenizer.json \
  --video data/video_smoke/videos/red_to_blue.mp4 \
  --prompt "What color is shown at the end?" \
  --frames 2 --frame-sampling uniform \
  --max-tokens 8 --greedy --safety none
```

See [the Phase 35 guide](docs/phase35_video.md) for fixture generation,
training, NExT-QA evaluation, supported formats, and memory controls.

### Understand Documents

Phase 36 renders PDFs or scanned page images through the same frozen vision
encoder, then adds learned or sinusoidal row/column patch positions. It does
not invoke OCR or an external table parser.

Video-era tokenizers and SafeTensors checkpoints need one deterministic
three-row expansion before document tuning:

```sh
target/release/aarambh-ai convert \
  --config configs/video_qa_smoke.toml \
  --input checkpoints/video_smoke/model.safetensors \
  --output checkpoints/document_smoke/model.safetensors \
  --tokenizer checkpoints/video_smoke/tokenizer.json \
  --output-tokenizer checkpoints/document_smoke/tokenizer.json \
  --upgrade-document-vocab

target/release/aarambh-ai infer \
  --config configs/document_qa_smoke_infer.toml \
  --model checkpoints/document_qa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/document_smoke/tokenizer.json \
  --document data/document_smoke/documents/red_invoice.pdf \
  --pages 1 \
  --prompt "What color fills the first page?" \
  --max-tokens 8 --greedy --safety none
```

Run `python3 scripts/phase36_make_document_smoke_fixture.py` to create the
four local PDFs, or `scripts/phase36_smoke.sh` for the complete migration,
two-step tuning, inference, and ANLS-evaluation workflow. See
[the Phase 36 guide](docs/phase36_document.md) for the JSONL schema,
resource limits, official dataset import, and commands.

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
Use `--tasks tool-chain --agent-max-steps 8` for scripted multi-call
response-path evaluation. The checked-in fixture exercises three ordered calls;
BFCL v1.3 response paths can be normalized with
`scripts/phase37_prepare_bfcl_multiturn.py`.

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

### Train With Native QAT

Run the checked-in CPU smoke or continue an exact floating-point checkpoint:

```sh
target/release/aarambh-ai train --config configs/qat_smoke.toml
target/release/aarambh-ai train --config configs/qat_tiny.toml
```

`[model.qat]` supports `int4`/`int8`, `export_aligned`, `per_tensor`, or
`per_output_channel` scaling, and explicit projection target classes. Norms,
embeddings, convolution kernels, and scalar recurrent parameters remain full
precision. QAT is active only in the training constructor; saved SafeTensors
remain full-precision master weights and use the existing GGUF exporter.

Compare post-training quantization against QAT under identical eval settings:

```sh
scripts/phase34_compare_qat.sh \
  configs/qat_tiny.toml \
  checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  checkpoints/qat_tiny/best/model.safetensors \
  data/eval \
  reports/qat_tiny
```

The report contains four scorecards and direction-normalized quantization
drop/recovery. A positive QAT result is measured from that report, not assumed.

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

# Multi-token auxiliary loss and checkpoint save
target/release/aarambh-ai train --config configs/mtp_smoke.toml

# Local teacher, scored-reference teacher, offline control, eval, and resume
scripts/phase33_smoke.sh

# Native INT4 fake quantization, STE backward, checkpoints, and cache metrics
scripts/phase34_smoke.sh

# Native H.264 decode, video vocabulary migration, tuning, inference, and eval
scripts/phase35_smoke.sh
```

For an MTP-trained checkpoint, `--speculative` needs no draft checkpoint:

```sh
target/release/aarambh-ai infer \
  --config configs/mtp_smoke.toml \
  --model checkpoints/mtp_smoke/step_000002/model.safetensors \
  --tokenizer checkpoints/mtp_smoke/tokenizer.json \
  --prompt "To be, or not to be" \
  --max-tokens 64 --greedy --speculative --stats
```

Checkpoint retrofit and comparison tooling:

- `scripts/phase29_prepare_hybrid_retrofit.sh`
- `scripts/phase29_benchmark_hybrid.sh`
- `scripts/phase30_prepare_dsa_retrofit.sh`
- `scripts/phase30_benchmark_dsa.sh`
- `scripts/phase31_sweep_moe.sh`
- `scripts/phase32_compare_training.sh`
- `scripts/phase32_benchmark_mtp.sh`
- `scripts/phase33_prepare_prompts.py`
- `scripts/phase33_compare_distillation.sh`
- `scripts/phase34_smoke.sh`
- `scripts/phase34_compare_qat.sh`
- `scripts/phase35_make_video_smoke_fixture.py`
- `scripts/phase35_smoke.sh`
- `scripts/phase36_smoke.sh`
- `scripts/phase37_smoke.sh`

The Phase 31 method and result contract are documented in
[docs/phase31_moe_sweep.md](docs/phase31_moe_sweep.md). Hardware benchmark
results are not claimed until the scripts have produced scorecards.
The MTP training, retrofit, and one-checkpoint speculation contract is in
[docs/phase32_mtp.md](docs/phase32_mtp.md).
The on-policy teacher, replay objective, offline control, and matched comparison
protocol are in
[docs/phase33_distillation_results.md](docs/phase33_distillation_results.md).
The native QAT policy, exact continuation contract, and four-way robustness
gate are in [docs/phase34_qat.md](docs/phase34_qat.md).
The video decoding, sampling, temporal fusion, migration, training, and eval
contract is in [docs/phase35_video.md](docs/phase35_video.md).

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

The workspace contains 18 internal library crates and one CLI package:

```text
aarambh-ai-core        Shared config, device, dtype, errors, and traits
aarambh-ai-tokenizer   BPE tokenizer and reserved special tokens
aarambh-ai-data        Datasets, preprocessing, sharding, and loaders
aarambh-ai-kernel      CPU SIMD and optional CUDA kernels
aarambh-ai-nn          Neural layers, attention, DeltaNet, DSA, MoE, and MTP
aarambh-ai-model       Full decoder model and cache integration
aarambh-ai-weights     SafeTensors, GGUF, conversion, and retrofit loading
aarambh-ai-quant       INT8/INT4, GPTQ, AWQ, QAT, and KV quantization
aarambh-ai-train       Optimizer, schedules, MTP loss, checkpoints, distributed train
aarambh-ai-finetune    Adapters, SFT, GRPO, DPO, VLM, and tool tuning
aarambh-ai-inference   Sampling, caching, thinking, MTP/external speculation, tools
aarambh-ai-agent       Bounded tool chains, exact state, and caller-result ingestion
aarambh-ai-safety      Input, output, streaming, PII, and audit policies
aarambh-ai-selflearn   Critique, replay, verifiers, and persistent update state
aarambh-ai-eval        Evaluation tasks, scorecards, and comparisons
aarambh-ai-vision      Image/video/document decode, preprocessing, temporal/layout fusion
aarambh-ai-distill     On-policy rollouts, teacher scoring, losses, and resume
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
| [ARCHITECTURE_V3.md](ARCHITECTURE_V3.md) | v3 hybrid attention, DSA, fine-grained MoE, MTP, and planned architecture |
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
| [docs/phase32_mtp.md](docs/phase32_mtp.md) | MTP training, retrofit, exact speculation, and benchmark method |
| [docs/phase33_distillation_results.md](docs/phase33_distillation_results.md) | On-policy distillation design, smoke proof, and comparison method |
| [docs/phase34_qat.md](docs/phase34_qat.md) | Native QAT configuration, continuation, export, and robustness validation |
| [docs/phase35_video.md](docs/phase35_video.md) | Video migration, decoding, tuning, inference, and NExT-QA evaluation |
| [docs/phase36_document.md](docs/phase36_document.md) | PDF/page ingestion, layout tuning, inference, and DocVQA ANLS evaluation |
| [docs/phase37_agent.md](docs/phase37_agent.md) | Tool-chain protocol, safety, context policy, SFT, and response-path evaluation |
| [RELEASE.md](RELEASE.md) | Source-release process and artifact policy |
| [CHANGELOG.md](CHANGELOG.md) | Versioned implementation history |

## Current Boundaries

- No pretrained model is included, so output quality depends entirely on your
  training data, update count, and tuning.
- GGUF tensors are dequantized when loaded by the current universal model path;
  smaller files do not yet imply fully quantized compute.
- QAT robustness improvements require a trained checkpoint and a positive
  four-way eval report; the source release does not claim unexecuted gains.
- MoE dispatch computes every routed expert and applies dense weights. Fine
  granularity changes capacity and routing but is not sparse grouped dispatch.
- Multi-GPU support is single-node data parallel training.
- Tool calls and chains are generated/orchestrated but never executed by the
  model runtime; callers remain responsible for authorization and execution.
- The server currently hosts one text model and one generated choice per
  request; vision and self-learning are CLI workflows.
- Video understanding is visual-only and currently accepts H.264 MP4 input;
  audio, other containers/codecs, server upload, and video self-learning are
  outside Phase 35.
- Document understanding is pixel-based: no OCR/table parser is bundled.
  Document server upload, self-learning, speculative decoding, and tool calling
  remain outside Phase 36.
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
