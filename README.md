# aarambh-ai

> **Sanskrit: *beginning* — A ground-up LLM in Rust**

[![CI](https://github.com/AarambhDevHub/aarambh-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AarambhDevHub/aarambh-ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg)](https://www.rust-lang.org)

A decoder-only transformer with four model scales, a three-level thinking engine, full training pipeline, quantisation (INT8/INT4/GGUF), LoRA/QLoRA fine-tuning, GRPO reinforcement learning, custom CUDA + SIMD kernels, safety guardrails, and a self-learning loop — all in one clean 14-crate Rust workspace.

**Inspired by:** LLaMA · Mistral · DeepSeek · GPT · Claude · Qwen · Gemma

---

## Features

| Capability | Status |
|---|---|
| BPE tokeniser (train, encode, decode, special tokens) | Phase 1 ✅ |
| Data pipeline (datasets, chunk & tokenize, DataLoader) | Phase 1 ✅ |
| Decoder-only transformer (RMSNorm, RoPE, GQA, SwiGLU) | Phase 2 ✅ |
| Four model scales: Tiny (25M) → Large (1.3B) | Phase 0 ✅ |
| Full model forward pass (embedding, blocks, LM head, KV cache path) | Phase 3 ✅ |
| SafeTensors model save/load | Phase 3 ✅ |
| Thinking engine: Low / Medium / High reasoning budgets | Phase 7 |
| Full training pipeline with AdamW, cosine schedule, checkpointing | Phase 5 |
| Quantisation: INT8, GPTQ INT4, AWQ INT4, GGUF, QAT | Phase 8 |
| LoRA, QLoRA, DoRA fine-tuning | Phase 9 |
| GRPO reinforcement learning | Phase 10 |
| Safety guardrails: input/output, PII, prompt injection | Phase 11 |
| Self-learning loop: online GRPO, replay buffer, critique | Phase 12 |
| Custom CUDA kernels: Flash Attention v2, fused RMSNorm, RoPE, SwiGLU | Phase 14 |
| CPU SIMD kernels: AVX2/FMA RMSNorm, AVX512 override, parallel attention via rayon | Phase 4 ✅ |
| CUDA kernel build prep and FFI stubs | Phase 4 ✅ |
| CLI binary with predict-view, streaming, thinking modes | Phase 6 |

---

## Quick Start

### Prerequisites

- Rust 1.80+ ([install via rustup](https://rustup.rs/))
- No GPU required for development (Tiny trains on any i3 laptop)

### Build & Test

```sh
git clone https://github.com/AarambhDevHub/aarambh-ai.git
cd aarambh-ai

# Check the entire workspace compiles
cargo check --workspace

# Run all tests
cargo test --workspace

# Build a release binary
cargo build --release

# Run the CLI
cargo run --release -- --help
```

---

## Architecture

```
aarambh-ai/
├── aarambh-ai-core/          ← Foundation types (config, device, dtype, error, traits)
├── aarambh-ai-tokenizer/     ← BPE tokeniser, vocab, special tokens
├── aarambh-ai-data/          ← Datasets, preprocessing, data loader
├── aarambh-ai-nn/            ← RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock
├── aarambh-ai-kernel/        ← Custom CUDA + CPU SIMD kernels
├── aarambh-ai-model/         ← Embedding, LM head, full model forward pass
├── aarambh-ai-weights/       ← SafeTensors I/O (GGUF + HuggingFace conversion planned)
├── aarambh-ai-quant/         ← INT8, GPTQ, AWQ, GGUF, KV cache quant
├── aarambh-ai-train/         ← Training loop, AdamW, cosine schedule, checkpointing
├── aarambh-ai-finetune/      ← LoRA, QLoRA, SFT, GRPO, verifiers
├── aarambh-ai-inference/     ← Inference engine, KV cache, sampler, streaming
├── aarambh-ai-safety/        ← Input/output guardrails, PII, audit
├── aarambh-ai-selflearn/     ← Self-learning loop, replay buffer, critique
└── aarambh-ai/               ← CLI binary (train, infer, finetune, quant, eval)
```

### Dependency Layers

```
Layer 0  aarambh-ai-core
Layer 1  aarambh-ai-tokenizer   aarambh-ai-data
Layer 2  aarambh-ai-nn          aarambh-ai-kernel
Layer 3  aarambh-ai-model       aarambh-ai-weights    aarambh-ai-quant
Layer 4  aarambh-ai-train       aarambh-ai-finetune
Layer 5  aarambh-ai-inference   aarambh-ai-safety     aarambh-ai-selflearn
Layer 6  aarambh-ai (binary)
```

Every crate depends only on crates in the same or lower layer. This is enforced by `Cargo.toml`.

---

## Model Scales

| Scale | Params | d_model | Layers | Heads | KV Heads | d_ffn | Max seq | rope_theta |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Tiny | 25M | 384 | 8 | 6 | 2 | 1,024 | 512 | 10,000 |
| Small | 117M | 768 | 12 | 12 | 4 | 2,688 | 1,024 | 10,000 |
| Medium | 360M | 1,024 | 24 | 16 | 8 | 3,392 | 2,048 | 500,000 |
| Large | 1.3B | 2,048 | 24 | 32 | 8 | 6,656 | 4,096 | 500,000 |

All scales share `vocab_size=32000`, `norm_eps=1e-5`, and weight-tied embeddings.

**Which scale to use:**

| Scale | Hardware | Best for |
|---|---|---|
| **Tiny** (25M) | i3 laptop | Full train + infer. Use for all development & debugging. |
| **Small** (117M) | Kaggle T4 (16 GB) | GPT-2 equivalent. Thinking engine testing. |
| **Medium** (360M) | Kaggle P100 / A100 | Real text quality. Instruction tuning. |
| **Large** (1.3B) | Kaggle A100 40 GB | Genuine reasoning in High thinking mode. |

---

## Core Concepts

### The Full Journey: Token → Output

```
User types: "The capital of India is"
       │
       ▼
┌─────────────────────────────┐
│  TOKENISER                  │
│  "The capital of India is"  │
│  → [464, 3139, 286, 4826, 318]
└─────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────┐
│  EMBEDDING TABLE            │
│  Each ID → d_model vector   │
│  Shape: [1, 5, d_model]     │
└─────────────┬───────────────┘
               │  × N_layers
               ▼
┌─────────────────────────────┐
│  TRANSFORMER BLOCK          │
│  ┌─ RMSNorm ──────────────┐ │
│  │  GQA + RoPE + mask     │ │
│  │  + KV cache (inference)│ │
│  └────────────────────────┘ │
│       x = x + attention     │  ← residual
│  ┌─ RMSNorm ──────────────┐ │
│  │  SwiGLU FFN             │ │
│  └────────────────────────┘ │
│       x = x + ffn           │  ← residual
└─────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────┐
│  FINAL RMSNorm + LM HEAD    │
│  [1, 5, d_model] → logits   │
└─────────────┬───────────────┘
               │
               ▼
┌─────────────────────────────┐
│  SAMPLER                    │
│  softmax → sample → token   │
└─────────────────────────────┘
```

### Thinking Engine

The model generates a `<think>` block before its answer, with budget enforcement:

| Mode | Budget | Best for |
|---|---|---|
| None | 0 tokens | Simple facts, maximum speed |
| Low | ≤ 256 tokens | One-step reasoning, basic Q&A |
| Medium | ≤ 1,024 tokens | Multi-step math, coding, logic |
| High | ≤ 4,096 tokens | Complex reasoning, planning |

---

## Project Structure

```
aarambh-ai/
├── Cargo.toml                   ← Workspace manifest (pinned deps)
├── ARCHITECTURE.md              ← System design and documentation
├── ROADMAP.md                   ← Phased delivery plan
├── SELF_LEARNING.md             ← Self-learning loop design
├── crates/
│   ├── aarambh-ai-core/         ← Foundation types
│   ├── aarambh-ai-tokenizer/    ← BPE tokeniser
│   ├── aarambh-ai-data/         ← Datasets and data loading
│   ├── aarambh-ai-nn/           ← Neural network primitives
│   ├── aarambh-ai-kernel/       ← Custom kernels
│   ├── aarambh-ai-model/        ← Full model assembly
│   ├── aarambh-ai-weights/      ← Weight serialisation
│   ├── aarambh-ai-quant/        ← Quantisation stack
│   ├── aarambh-ai-train/        ← Training loop
│   ├── aarambh-ai-finetune/     ← Fine-tuning (LoRA, etc.)
│   ├── aarambh-ai-inference/    ← Inference engine
│   ├── aarambh-ai-safety/       ← Safety guardrails
│   └── aarambh-ai-selflearn/    ← Self-learning loop
├── aarambh-ai/                  ← CLI binary
├── .github/                     ← CI, issue templates, PR template
├── LICENSE                      ← Apache 2.0
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
└── SECURITY.md
```

---

## Development Roadmap

| Phase | What | Hardware | Status |
|---|---|---|---|
| 0 | Workspace + core types | i3 | ✅ |
| 1 | Tokeniser + data pipeline | i3 | ✅ |
| 2 | Neural network primitives | i3 | ✅ |
| 3 | Full model forward pass | i3 | ✅ |
| 4 | Custom kernels (CPU SIMD + CUDA stubs) | i3 + GPU | ✅ |
| 5 | Training loop — Tiny trains! | i3 | ⬜ |
| 6 | Inference engine + CLI | i3 | ⬜ |
| 7 | Thinking engine | i3 | ⬜ |
| 8 | Quantisation stack | i3 | ⬜ |
| 9 | Fine-tuning (LoRA, QLoRA, SFT) | i3 + GPU | ⬜ |
| 10 | GRPO reinforcement learning | GPU | ⬜ |
| 11 | Safety layer | i3 | ⬜ |
| 12 | Self-learning loop | i3 + GPU | ⬜ |
| 13 | GPU scale-up (Small → Large) | GPU | ⬜ |
| 14 | Flash Attention CUDA kernels | GPU | ⬜ |
| 15 | Production release v1.0 | all | ⬜ |

See [ROADMAP.md](ROADMAP.md) for the full phased delivery plan with tests and milestones.

---

## Development Checks

```sh
cargo check --workspace
cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo fmt --check
cargo doc --workspace --no-deps
```

### Kernel Benchmarks

```sh
cargo bench -p aarambh-ai-kernel
```

Phase 4 uses stable CPU intrinsics with cached AVX2/FMA, AVX512, and scalar
dispatch. The default prefers AVX2/FMA on this CPU; set `AARAMBH_SIMD_FORCE=avx512`
to force AVX512 when it wins on another machine. CUDA files are compiled only
when `nvcc` is installed; otherwise the build emits a warning and keeps the
Candle fallback path.

---

## Documentation

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete design document, layer-by-layer explanation, data flow, thinking engine, quantisation, fine-tuning, safety, self-learning |
| [ROADMAP.md](ROADMAP.md) | Step-by-step build plan with tasks, tests, and milestones for all 15 phases |
| [SELF_LEARNING.md](SELF_LEARNING.md) | Deep dive into the self-learning loop: online GRPO, replay buffer, self-critique, CPU vs GPU modes |

---

## Citation

If you use aarambh-ai in your research, please cite it as follows:

```bibtex
@software{aarambh_ai_2026,
  title        = {aarambh-ai: A Ground-Up LLM in Rust},
  author       = {Aarambh Dev Hub},
  year         = {2026},
  url          = {https://github.com/AarambhDevHub/aarambh-ai},
  version      = {0.0.1},
  license      = {Apache-2.0},
}
```

---

## Support

- Star the repo on [GitHub](https://github.com/AarambhDevHub/aarambh-ai)
- Open [issues](https://github.com/AarambhDevHub/aarambh-ai/issues) for reproducible bugs and clear feature requests
- Join the discussion on [Discord](https://discord.gg/aarambhdevhub)
- Report security vulnerabilities via [SECURITY.md](SECURITY.md)
- Support development through [Buy Me a Coffee](https://buymeacoffee.com/aarambhdevhub) or [GitHub Sponsors](https://github.com/sponsors/aarambh-darshan)

---

## License

Apache 2.0 © [AarambhDevHub](https://github.com/AarambhDevHub)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
