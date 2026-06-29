# ARCHITECTURE.md — aarambh-ai

> A modern, from-scratch LLM in Rust using `candle`. Decoder-only transformer with
> thinking capability, four model scales, quantisation, fine-tuning, safety guardrails,
> custom kernels, and a self-learning loop — all in one clean 14-crate workspace.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [Dependency Versions & Toolchain](#3-dependency-versions--toolchain)
4. [Complete Workspace — 14 Crates](#4-complete-workspace--14-crates)
5. [Model Scales](#5-model-scales)
6. [The Full Journey: Token → Output](#6-the-full-journey-token--output)
   - 6.1 Tokenisation
   - 6.2 Embedding Layer
   - 6.3 Rotary Positional Embedding (RoPE)
   - 6.4 RMSNorm
   - 6.5 Grouped-Query Attention (GQA)
   - 6.6 SwiGLU Feed-Forward Network
   - 6.7 Residual Connections & Pre-Norm Layout
   - 6.8 Output LM Head
7. [Thinking Engine](#7-thinking-engine)
8. [KV Cache & Inference](#8-kv-cache--inference)
9. [Training Pipeline](#9-training-pipeline)
10. [Custom Kernels](#10-custom-kernels-aarambh-ai-kernel)
11. [Quantisation](#11-quantisation-aarambh-ai-quant)
12. [Fine-Tuning](#12-fine-tuning-aarambh-ai-finetune)
13. [Safety Layer](#13-safety-layer-aarambh-ai-safety)
14. [Self-Learning Loop](#14-self-learning-loop-aarambh-ai-selflearn)
    - 14.1 Overview & Design — built in Phase 12, before GPU scale-up
    - 14.2 Online GRPO (uses deterministic verifiers)
    - 14.3 Experience Replay Buffer
    - 14.4 Self-Critique Loop (free function, replay-only)
    - 14.5 CPU-safe Mode (i3)
    - 14.6 Full Loop Flow
    - 14.7 SelfLearnConfig
15. [Crate-by-Crate Reference](#15-crate-by-crate-reference)
16. [Data Flow Across the Workspace](#16-data-flow-across-the-workspace)
17. [Memory & Compute Estimates](#17-memory--compute-estimates)
18. [Hardware Strategy](#18-hardware-strategy)

---

## 1. Project Overview

**aarambh-ai** (Sanskrit: *beginning*) is a ground-up LLM written entirely in Rust.
It is not a wrapper around PyTorch or any Python library. Every layer, every training
loop, every kernel is implemented from scratch using `candle-core` and `candle-nn`
as the tensor backend.

### What it is

- A decoder-only transformer (same family as GPT, LLaMA, Mistral, Qwen)
- Four model sizes: Tiny (25M) → Small (117M) → Medium (360M) → Large (1.3B)
- Three-level thinking engine: Low / Medium / High reasoning depth
- Full training pipeline: pretraining → SFT → GRPO reinforcement learning
- Quantisation: INT8, GPTQ INT4, AWQ INT4, GGUF Q4_K_M
- Quantisation-Aware Training (QAT) for higher-quality INT4 checkpoints
- Fine-tuning: LoRA, QLoRA, instruction tuning, thinking tuning
- Custom kernels: Flash Attention v2, fused RMSNorm, fused RoPE, fused SwiGLU
- Weight conversion: load HuggingFace-format checkpoints directly (pragmatic slicing)
- Safety: input guardrails, output guardrails, PII protection, audit logging
- Self-learning: Online GRPO (with deterministic verifiers), experience replay buffer, self-critique loop (replay-only) — model improves from its own outputs

### What makes it different

Every other LLM tutorial or project either wraps Python tools in Rust or skips
the hard parts. aarambh-ai builds everything: the BPE tokeniser, the attention
mechanism, the optimiser, the quantisation algorithm, the LoRA injector, the
safety guard — all in one workspace, all in Rust, all explained in this document.

---

## 2. Design Philosophy

| Goal | Decision |
|---|---|
| Modern architecture only | RMSNorm, SwiGLU, RoPE, GQA, pre-norm — no legacy components |
| One codebase, all scales | `ModelConfig` drives every dimension — zero code duplication |
| Thinking out of the box | `<think>` / `</think>` tokens, budget enforcement, three modes |
| Quantisation-first | Every scale has a quantised variant; INT4 is the deployment target |
| Efficient fine-tuning | LoRA + QLoRA so Small can be fine-tuned on an i3 laptop |
| Safety as first-class | `aarambh-ai-safety` wraps inference; not an afterthought |
| Self-learning loop | Model improves from its own outputs — no human labels needed after SFT |
| CPU-first development | Tiny trains and generates on i3/8GB; no GPU required for dev |
| Clean crate boundaries | No sideways dependencies; strict layering enforced by `Cargo.toml` |
| HuggingFace compatible | SafeTensors checkpoints, GGUF export, HF weight conversion (pragmatic) |
| Toolchain pinned | Exact versions documented — no "works on my machine" surprises |

---

## 3. Dependency Versions & Toolchain

> **Pin these versions.** Candle's public API has changed across minor releases.
> Using a different version from what is listed here may break compilation.
> Run `rustup update stable` before starting. Phase 4 SIMD uses stable intrinsics.

### Rust Toolchain

```
stable:   rustup override set stable        ← default for all phases
```

> **Phase 4 note:** `aarambh-ai-kernel` uses stable `std::arch` intrinsics with
> cached AVX2/FMA, AVX512, AVX2, and scalar dispatch. No nightly toolchain is
> required.

### Verified Dependency Versions

```toml
candle-core    = "0.10"          # confirmed latest stable on crates.io
candle-nn      = "0.10"
tokenizers     = "0.21"          # HuggingFace tokenizers Rust crate (stable API)
safetensors    = "0.8"
thiserror      = "2"
serde          = { version = "1", features = ["derive"] }
serde_json     = "1"
tokio          = { version = "1", features = ["full"] }
clap           = { version = "4", features = ["derive"] }
tracing        = "0.1"
tracing-subscriber = "0.3"
rayon          = "1"
cc             = "1"
which          = "6"
anyhow         = "1"
criterion      = "0.5"
```

### Tokenizer Strategy (important)

The `tokenizers` crate (HuggingFace) is used for **both loading and training**
BPE tokenizer files. `aarambh-ai-tokenizer` wraps this for `from_pretrained()` and
also uses it for `train()` (delegates the heavy lifting). The pure-Rust `BpeTokenizer`
implements `encode()` and `decode()` from the merge rules, which is fast and
dependency-free at runtime.

The two are complementary:
- `BpeTokenizer::from_pretrained(path)` → uses `tokenizers` crate internally, loads
  a `tokenizer.json` produced by HuggingFace, OpenAI GPT-2, or similar
- `BpeTokenizer::train(corpus_path, vocab_size)` → delegates to `tokenizers::trainers::BpeTrainer`,
  reserves IDs 0..6 for project special tokens, then saves and loads the result
  via `from_pretrained()`
- `encode()`/`decode()` are pure-Rust (no external deps at inference time)

The first seven token IDs are architectural, not learned from the corpus:
`<|endoftext|>` = 0, `<|pad|>` = 1, `<|bos|>` = 2, `<think>` = 3,
`</think>` = 4, `<|user|>` = 5, and `<|assistant|>` = 6. Training reuses an
existing tokenizer only if these IDs validate; inference rejects tokenizers that
do not validate, because EOS stopping and Phase 7 thinking control depend on
stable IDs.

**For Phase 1 tests:** download GPT-2's `tokenizer.json` from HuggingFace and place it
at `tests/fixtures/tokenizer.json`. This solves the chicken-and-egg problem — you
have a valid tokenizer for tests before training your own on a custom corpus.

```bash
# Download GPT-2 tokenizer fixture (run once before Phase 1 tests)
curl -L https://huggingface.co/gpt2/resolve/main/tokenizer.json \
     -o crates/aarambh-ai-tokenizer/tests/fixtures/tokenizer.json
```

---

## 4. Complete Workspace — 14 Crates

```
aarambh-ai/
├── Cargo.toml                        ← [workspace] manifest, shared dependencies
├── ARCHITECTURE.md                   ← this file
├── ROADMAP.md                        ← phased delivery plan
│
├── crates/
│   │
│   ├── aarambh-ai-core/              ← LAYER 0: Foundation types (no ML deps)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── config.rs             ← ModelConfig (4 presets), TrainConfig (incl. eval_steps)
│   │       ├── device.rs             ← Device enum, best_available()
│   │       ├── dtype.rs              ← DType (F32/F16/BF16), Precision
│   │       ├── error.rs              ← AarambhError, Result<T>
│   │       └── traits.rs             ← Forward, Saveable, Loadable, TokenizerLike
│   │
│   ├── aarambh-ai-tokenizer/         ← LAYER 1: Text → token IDs
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── bpe.rs                ← BPE merge algorithm (load + encode/decode)
│   │       ├── vocab.rs              ← Vocab struct, token↔ID maps
│   │       └── special.rs            ← <think>, </think>, <|user|>, etc.
│   │
│   ├── aarambh-ai-data/              ← LAYER 1: Raw text → batched tensors
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dataset.rs            ← TextDataset, JsonlDataset traits
│   │       ├── loader.rs             ← DataLoader, batch collation, padding
│   │       └── preprocess.rs         ← chunking, shift-by-1 label creation
│   │
│   ├── aarambh-ai-nn/                ← LAYER 2: Neural network primitives
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── norm.rs               ← RMSNorm
│   │       ├── rope.rs               ← RopeCache, apply_rope()
│   │       ├── attention.rs          ← GroupedQueryAttention
│   │       ├── ffn.rs                ← SwiGluFfn
│   │       └── block.rs              ← TransformerBlock
│   │
│   ├── aarambh-ai-kernel/            ← LAYER 2: Custom kernels (CUDA + CPU SIMD)
│   │   ├── build.rs                  ← NVCC detection, .cu compilation
│   │   ├── kernels/
│   │   │   ├── flash_attention.cu    ← Flash Attention v2 forward
│   │   │   ├── flash_attn_bwd.cu     ← Flash Attention v2 backward
│   │   │   ├── rms_norm_fused.cu     ← single-pass fused RMSNorm
│   │   │   ├── rope_apply.cu         ← fused RoPE on Q and K
│   │   │   └── swiglu_fused.cu       ← fused gate×up→swish
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── dispatch.rs           ← runtime: CUDA kernel vs candle fallback
│   │       ├── flash_attn.rs         ← Rust FFI for flash_attention.cu
│   │       ├── fused_norm.rs         ← Rust FFI for rms_norm_fused.cu
│   │       ├── fused_rope.rs         ← Rust FFI for rope_apply.cu
│   │       ├── fused_ffn.rs          ← Rust FFI for swiglu_fused.cu
│   │       └── cpu/
│   │           ├── simd_norm.rs      ← Stable AVX2/FMA + AVX512 SIMD RMSNorm
│   │           └── parallel_attn.rs  ← rayon parallel attention heads (stable)
│   │
│   ├── aarambh-ai-model/             ← LAYER 3: Full model assembly
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── embedding.rs          ← TokenEmbedding (weight-tied)
│   │       ├── head.rs               ← LM head: linear → logits
│   │       └── model.rs              ← AarambhModel: embed + N×block + head
│   │
│   ├── aarambh-ai-weights/           ← LAYER 3: Serialisation I/O
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── safetensors.rs        ← save_model(), load_model()
│   │       ├── gguf.rs               ← GGUF reader/writer (Q4_K_M, Q5_K_M, Q8_0)
│   │       └── convert.rs            ← HuggingFace → aarambh-ai weight format
│   │                                    (renames keys, slices GQA tensors strictly,
│   │                                     handles tied vs untied LM head)
│   │
│   ├── aarambh-ai-quant/             ← LAYER 3: Quantisation stack
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── absmax.rs             ← INT8 absmax quantisation
│   │       ├── dequant.rs            ← runtime dequantise → F32 or BF16
│   │       ├── calibrate.rs          ← calibration dataset runner (128 samples)
│   │       ├── gptq.rs               ← GPTQ INT4 (Hessian + Cholesky inversion)
│   │       ├── awq.rs                ← AWQ INT4 (activation-aware, no inversion)
│   │       ├── gguf_quant.rs         ← Q4_K_M, Q5_K_M, Q8_0 block schemes
│   │       ├── kv_quant.rs           ← KV cache INT8 quantisation
│   │       └── qat.rs                ← Quantisation-Aware Training (fake quant nodes)
│   │
│   ├── aarambh-ai-train/             ← LAYER 4: Training loop
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── trainer.rs            ← Trainer, train_step(), train_epoch()
│   │       ├── loss.rs               ← cross_entropy_loss(), label_smoothing
│   │       ├── optim.rs              ← AdamW (β₁=0.9, β₂=0.95, ε=1e-8, λ=0.1)
│   │       ├── schedule.rs           ← cosine LR + warmup
│   │       └── checkpoint.rs         ← save/load training state
│   │
│   ├── aarambh-ai-finetune/          ← LAYER 4: Fine-tuning stack
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── lora.rs               ← LoraLayer, inject_lora(), merge_lora()
│   │       ├── qlora.rs              ← QLoRA: INT4 base + BF16 adapters
│   │       │                            (dequant on every forward for autograd)
│   │       ├── sft.rs                ← SftTrainer, chat template, loss masking
│   │       ├── grpo.rs               ← GRPO: group sampling, advantage, KL
│   │       │                            (uses deterministic verifiers only)
│   │       ├── adapter.rs            ← save/load LoRA adapter weights only
│   │       └── verifier.rs           ← MathVerifier, FormatVerifier, CodeVerifier
│   │
│   ├── aarambh-ai-inference/         ← LAYER 5: Inference engine
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── engine.rs             ← InferenceEngine, generate(), prefill+decode
│   │       ├── kvcache.rs            ← KVCache per layer, sequence growth
│   │       ├── sampler.rs            ← Greedy, TopK, TopP, Temperature
│   │       ├── thinking.rs           ← ThinkingMode, ThinkingController, budget
│   │       └── stream.rs             ← token-by-token mpsc channel streaming
│   │
│   ├── aarambh-ai-safety/            ← LAYER 5: Safety guardrails
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── guard.rs              ← SafetyGuard wraps InferenceEngine
│   │       ├── policy.rs             ← SafetyPolicy: which checks, thresholds
│   │       ├── verdict.rs            ← Allow / Block / Redact / Regenerate
│   │       ├── input/
│   │       │   ├── mod.rs
│   │       │   ├── injection.rs      ← prompt injection detection
│   │       │   ├── jailbreak.rs      ← jailbreak patterns + encoding tricks
│   │       │   └── pii.rs            ← email, phone, SSN, credit card, API keys
│   │       └── output/
│   │           ├── mod.rs
│   │           ├── toxicity.rs       ← 5-category toxicity scorer
│   │           ├── pii_redact.rs     ← redact PII in model output
│   │           └── audit.rs          ← SafetyEvent logging → safety_audit.jsonl
│   │
│   └── aarambh-ai-selflearn/         ← LAYER 5: Self-learning loop
│       └── src/
│           ├── lib.rs
│           ├── loop.rs               ← SelfLearnLoop (owns OnlineGrpo, Replay)
│           ├── config.rs             ← SelfLearnConfig (mode, thresholds, budget)
│           ├── online_grpo.rs        ← generate N completions → score (deterministic) → train online
│           ├── replay.rs             ← ReplayBuffer: store, sample, evict
│           ├── critique.rs           ← critique_response() free function (replay-only)
│           └── metrics.rs            ← track improvement per topic over time
│
└── aarambh-ai/                       ← LAYER 6: CLI binary (published to crates.io)
    └── src/
        ├── main.rs
        ├── cmd/
        │   ├── train.rs              ← `aarambh-ai train`
        │   ├── infer.rs              ← `aarambh-ai infer`
        │   ├── finetune.rs           ← `aarambh-ai finetune sft / grpo`
        │   ├── quantise.rs           ← `aarambh-ai quantise`
        │   ├── convert.rs            ← `aarambh-ai convert` (HF → aarambh format)
        │   └── eval.rs               ← `aarambh-ai eval`
        └── ui/
            └── predict_view.rs       ← coloured next-token probability display
```

### Dependency Layers (strict, no sideways deps)

```
Layer 0  aarambh-ai-core
Layer 1  aarambh-ai-tokenizer   aarambh-ai-data
Layer 2  aarambh-ai-nn          aarambh-ai-kernel
Layer 3  aarambh-ai-model       aarambh-ai-weights    aarambh-ai-quant
Layer 4  aarambh-ai-train       aarambh-ai-finetune
Layer 5  aarambh-ai-inference   aarambh-ai-safety     aarambh-ai-selflearn
Layer 6  aarambh-ai (binary)
```

Every crate may only depend on crates in the same or lower layer.
This is enforced by `Cargo.toml` — the compiler will catch any violation.

### Per-Crate `Cargo.toml` Dependencies (quick reference)

When you `cargo new` each crate, add exactly these workspace deps to its `Cargo.toml`:

| Crate | `[dependencies]` to add |
|---|---|
| `aarambh-ai-core` | `candle-core`, `serde`, `thiserror`, `tracing` |
| `aarambh-ai-tokenizer` | `aarambh-ai-core`, `tokenizers`, `serde_json` |
| `aarambh-ai-data` | `aarambh-ai-core`, `aarambh-ai-tokenizer`, `candle-core`, `serde_json`, `rayon` |
| `aarambh-ai-nn` | `aarambh-ai-core`, `aarambh-ai-kernel`, `candle-core`, `candle-nn` |
| `aarambh-ai-kernel` | `aarambh-ai-core`, `candle-core`, `candle-nn`, `rayon`, `cc`, `which`, `criterion` |
| `aarambh-ai-model` | `aarambh-ai-core`, `aarambh-ai-nn`, `aarambh-ai-kernel`, `candle-core`, `candle-nn` |
| `aarambh-ai-weights` | `aarambh-ai-core`, `aarambh-ai-model`, `candle-core`, `safetensors`, `serde_json` |
| `aarambh-ai-quant` | `aarambh-ai-core`, `aarambh-ai-model`, `aarambh-ai-weights`, `candle-core` |
| `aarambh-ai-train` | `aarambh-ai-core`, `aarambh-ai-model`, `aarambh-ai-data`, `aarambh-ai-weights`, `candle-core`, `candle-nn` |
| `aarambh-ai-finetune` | `aarambh-ai-core`, `aarambh-ai-model`, `aarambh-ai-train`, `aarambh-ai-quant`, `candle-core`, `candle-nn` |
| `aarambh-ai-inference` | `aarambh-ai-core`, `aarambh-ai-model`, `aarambh-ai-weights`, `candle-core`, `tokio` |
| `aarambh-ai-safety` | `aarambh-ai-core`, `aarambh-ai-inference`, `serde_json` |
| `aarambh-ai-selflearn` | `aarambh-ai-core`, `aarambh-ai-inference`, `aarambh-ai-finetune`, `candle-core`, `serde_json` |
| `aarambh-ai` (binary) | all 13 library crates, `clap`, `anyhow`, `tokio`, `tracing-subscriber` |

All deps use the `workspace = true` key, e.g.:
```toml
# crates/aarambh-ai-core/Cargo.toml
[dependencies]
candle-core  = { workspace = true }
serde        = { workspace = true }
thiserror    = { workspace = true }
tracing      = { workspace = true }
```

---

## 5. Model Scales

All four scales share the **identical code**. Only the numbers in `ModelConfig` differ.

```
┌──────────┬──────────┬──────────┬──────────┬──────────┬────────────┬──────────┬─────────────┐
│  Scale   │  Params  │ d_model  │ N_layers │ N_heads  │ N_kv_heads │ d_ffn    │ max_seq_len │
├──────────┼──────────┼──────────┼──────────┼──────────┼────────────┼──────────┼─────────────┤
│ Tiny     │   25M    │    384   │     8    │    6     │     2      │  1 024   │     512     │
│ Small    │  117M    │    768   │    12    │   12     │     4      │  2 688   │   1 024     │
│ Medium   │  360M    │  1 024   │    24    │   16     │     8      │  3 392   │   2 048     │
│ Large    │  1.3B    │  2 048   │    24    │   32     │     8      │  6 656   │   4 096     │
└──────────┴──────────┴──────────┴──────────┴──────────┴────────────┴──────────┴─────────────┘

vocab_size  : 32 000  (all scales)
head_dim    : 64  (all scales — d_model / N_heads is always 64)
rope_theta  : 10 000.0  (Tiny / Small)  |  500 000.0  (Medium / Large)
norm_eps    : 1e-5  (all scales)
tie_weights : true  (embedding ↔ LM head shared, all scales)
```

> **Param count verification (all exact):**
> Tiny: `embed(32000×384) + 8×(wq(384²) + wo(384²) + wk+wv(2×384×128) + ffn(3×384×1024) + norms(768)) + final_norm(384) = 24,877,440 ≈ 25M ✓`
> All four configs verified. `head_dim = d_model / N_heads = 64` for every scale.

**Which scale to use:**

- **Tiny (25M)** — your i3 laptop. Full train + infer. Use for all development & debugging.
  Produces coherent English after ~2K training steps on Shakespeare.
- **Small (117M)** — Kaggle T4 (16 GB). GPT-2 equivalent. Use for testing thinking engine.
- **Medium (360M)** — Kaggle P100 / A100. Real text quality. Instruction tuning meaningful.
- **Large (1.3B)** — Kaggle A100 40 GB. Genuine reasoning capability in High thinking mode.

---

## 6. The Full Journey: Token → Output

This section traces exactly what happens from "user types a prompt" to "model outputs
the next predicted token". Every operation is explained in plain language, with the
Rust types involved.

```
User types: "The capital of India is"
      │
      ▼
┌─────────────────────────────────────────┐
│  TOKENISER (aarambh-ai-tokenizer)       │
│  "The capital of India is"              │
│  → [464, 3139, 286, 4826, 318]          │  ← Vec<u32> token IDs
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  EMBEDDING TABLE (aarambh-ai-model)     │
│  Each ID → d_model-dim float vector     │
│  Tensor shape: [1, 5, 384]              │  ← (batch, seq_len, d_model)
└────────────────────┬────────────────────┘
                     │
                     ▼  repeated × N_layers
┌─────────────────────────────────────────┐
│  TRANSFORMER BLOCK (aarambh-ai-nn)      │
│                                         │
│  ┌──── RMSNorm(x) ──────────────────┐   │
│  │   GroupedQueryAttention           │   │
│  │   + RoPE positions                │   │
│  │   + causal mask                   │   │
│  │   + KV cache (inference only)     │   │
│  └───────────────────────────────────┘   │
│       x = x + attention_output           │  ← residual
│                                         │
│  ┌──── RMSNorm(x) ──────────────────┐   │
│  │   SwiGLU Feed-Forward             │   │
│  │   (gate × up → swish → down)      │   │
│  └───────────────────────────────────┘   │
│       x = x + ffn_output                 │  ← residual
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  FINAL RMSNorm + LM HEAD                │
│  [1, 5, 384] → [1, 5, 32000]            │  ← logits over vocabulary
│  Take last position: [1, 32000]          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  SAMPLER (aarambh-ai-inference)         │
│  softmax → probabilities → sample       │
│  → token ID → decode → "New"            │  ← predicted next token
└─────────────────────────────────────────┘
```

### 6.1 Tokenisation

The tokeniser converts raw text into a sequence of integer token IDs using
**Byte Pair Encoding (BPE)** — the same algorithm used in GPT-2, LLaMA, and Mistral.

**How BPE works:**
1. Start: every UTF-8 byte is its own token
2. Count the most frequent adjacent token pair in the training corpus
3. Merge that pair into one new token; record the merge rule
4. Repeat until vocabulary reaches `vocab_size` (32,000)

The learned merge table is saved as the tokeniser file. At inference, encoding
applies merge rules greedily to produce the final token ID sequence.

**Special tokens** (`aarambh-ai-tokenizer/src/special.rs`):
```
<|endoftext|>   ID 0   end of document
<|pad|>         ID 1   padding (batch alignment)
<|bos|>         ID 2   beginning of sequence
<think>         ID 3   open thinking scratchpad
</think>        ID 4   close thinking scratchpad, enter answer mode
<|user|>        ID 5   user turn marker (chat format)
<|assistant|>   ID 6   assistant turn marker
```

### 6.2 Embedding Layer

The embedding table is a weight matrix of shape `[32000, d_model]`.
Looking up token ID 464 means taking row 464 — a `d_model`-dimensional float vector.
For a 5-token sequence we produce a `[5, d_model]` tensor.

**Weight tying:** The embedding table and the LM head's weight are the **same tensor**
(`tie_embeddings = true`). This halves the parameter count for those layers and
improves quality — used by GPT-2, LLaMA-2/3, Qwen, Gemma.

### 6.3 Rotary Positional Embedding (RoPE)

Classical transformers inject position via a separate positional embedding added to
the token embedding. RoPE does it differently: it **rotates** the query and key
vectors inside attention using the position index.

**Rotation frequencies** (precomputed, cached):
```
θᵢ = rope_theta ^ (−2i / d_head)   for i = 0, 1, ..., d_head/2 − 1
```

**For each token at position m, each Q/K pair (x₀, x₁) becomes:**
```
x₀' = x₀ · cos(m·θᵢ) − x₁ · sin(m·θᵢ)
x₁' = x₀ · sin(m·θᵢ) + x₁ · cos(m·θᵢ)
```

**Why RoPE is better than additive PE:**
- Encodes *relative* position naturally — QᵢᵀKⱼ depends only on i−j
- Preserves vector magnitude (rotation, not scaling)
- Generalises to longer sequences than seen in training
- Zero extra parameters

The `RopeCache` in `aarambh-ai-nn/src/rope.rs` precomputes all cos/sin values for
`max_seq_len` positions at model init. Applied to Q and K before every attention call.

### 6.4 RMSNorm

Original transformers used LayerNorm: subtract mean, divide by std. **RMSNorm**
simplifies this — only divide by the root mean square, skip mean subtraction:

```
RMS(x)      = sqrt( mean(x²) + ε )
RMSNorm(x)  = (x / RMS(x)) * γ
```

- `ε = 1e-5` prevents division by zero
- `γ ∈ ℝᵈ` is a learned per-dimension scale, initialised to 1

**Why use it:** Faster (one fewer pass), more numerically stable, same or better
quality than LayerNorm. Used in LLaMA 1/2/3, Mistral, Qwen, Gemma, Phi-3.

**Placement:** Pre-norm — normalise *before* the sub-layer (attention or FFN), not
after. This is the key difference from original Transformers (Xiong et al., 2020).

### 6.5 Grouped-Query Attention (GQA)

Standard Multi-Head Attention (MHA) has one Q, K, V head per attention head.
**GQA** groups multiple Q heads to share one K and V head.

```
MHA:  Q[12]  K[12]  V[12]   → 12 separate attention computations
GQA:  Q[12]  K[4]   V[4]    → Q heads 0,1,2 share K[0],V[0] etc.
MQA:  Q[12]  K[1]   V[1]    → all Q heads share one K,V (extreme)
```

**Scaled dot-product attention (per head group):**
```
scores = softmax( Q · Kᵀ / sqrt(d_head) + causal_mask )
output = scores · V
```

The causal mask is lower-triangular (−∞ above diagonal): position i can only
attend to positions 0..=i. This is what makes generation autoregressive.

**Why GQA matters for inference:** The KV cache stores K and V for all past
tokens. For Small (12 Q heads, 4 KV heads) the cache is 3× smaller. For Large
(32 Q heads, 8 KV heads) it is 4× smaller — critical for long sequences.

### 6.6 SwiGLU Feed-Forward Network

Original transformer FFN: `max(0, xW₁)W₂` — two linear layers with ReLU.
Modern models use **SwiGLU**:

```
FFN(x) = ( Swish(x · W_gate) ⊙ (x · W_up) ) · W_down

Swish(z) = z · sigmoid(z) = z / (1 + e^−z)
⊙ = element-wise multiply
```

Three projection matrices:
- `W_gate [d_model, d_ffn]` — controls which features pass through
- `W_up   [d_model, d_ffn]` — the value being gated
- `W_down [d_ffn, d_model]` — projects back to model dimension

**Why SwiGLU:** The gating mechanism selectively amplifies or suppresses features.
Consistently better perplexity than ReLU/GeLU at same parameter count. Used in
LLaMA 1/2/3, PaLM, Gemma, Mistral, Qwen, Phi.

Note: `d_ffn` is chosen per the model scale table to hit each scale's exact parameter
count target. For Tiny, d_ffn=1024 = (8/3)×d_model (≈2.67×), making the three SwiGLU
matrices parameter-equivalent to a standard 2-matrix FFN at 4×d_model. For larger
scales, d_ffn is set to fill the parameter budget while keeping the ratio in the
2.67–3.5×d_model range.

### 6.7 Residual Connections & Pre-Norm Layout

Every transformer block applies two residual connections:

```
residual = x
x = RMSNorm(x)
x = GroupedQueryAttention(x, rope, mask, kv_cache)
x = residual + x               ← first residual

residual = x
x = RMSNorm(x)
x = SwiGluFfn(x)
x = residual + x               ← second residual
```

**Why residuals work:** Without them, gradients vanish in deep networks. With them,
the gradient can flow directly from the loss back to the embedding table through the
identity path, regardless of how many layers exist.

**Pre-norm:** Normalising *before* the sub-layer keeps activation scales stable
across all N layers. Post-norm (original Transformer) requires careful warmup
schedules to avoid divergence in deep networks; pre-norm does not.

### 6.8 Output LM Head

After all N blocks and a final RMSNorm, we take the last token position:
```
h ∈ ℝ^d_model   (the hidden state at the last position)
```

The LM head is a linear layer (no bias), weight-tied to the embedding:
```
logits = h · Eᵀ ∈ ℝ^vocab_size      (e.g. [1, 32000])
```

Applying softmax gives a probability distribution over all 32,000 tokens.
The token with the highest probability is the model's "best guess" for what
comes next. Sampling strategies (Top-P, temperature) control how we pick from
this distribution — see Section 8.

---

## 7. Thinking Engine

### 7.1 What Thinking Means

Modern reasoning models (OpenAI o3, DeepSeek-R1, Claude Extended Thinking) generate
an internal chain-of-thought *before* their final answer. The model "thinks on paper":
exploring the problem, checking its reasoning, backtracking — then commits to an answer.

In aarambh-ai, this is implemented via **budget-controlled CoT tokens**. The model
generates a `<think>` block (hidden or collapsible for the user), then produces the
final answer. The budget limits how many tokens the model can spend thinking.

### 7.2 Thinking Modes

```
┌───────────────┬─────────────────┬────────────────────────────────────────────┐
│  Mode         │  Token Budget   │  Best for                                  │
├───────────────┼─────────────────┼────────────────────────────────────────────┤
│  None         │  0              │  Simple facts, maximum speed               │
│  Low          │  ≤ 256 tokens   │  One-step reasoning, basic Q&A             │
│  Medium       │  ≤ 1 024 tokens │  Multi-step math, coding, logic            │
│  High         │  ≤ 4 096 tokens │  Complex reasoning, planning, hard problems│
└───────────────┴─────────────────┴────────────────────────────────────────────┘
```

### 7.3 How It Works at Inference

```
User: "What is 17 × 23?"

Step 1: System injects <|user|>\nWhat is 17 × 23?\n<|assistant|>
Step 2: ThinkingController forces <think> as first token (if mode != None)
Step 3: Model generates freely inside the thinking block:
          "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391"
Step 4: If tokens_used >= budget → force inject </think>
        OR model naturally produces </think>
Step 5: Model generates the final answer:
          "The answer is 391."
Step 6: Stop at <|endoftext|> or max_new_tokens
```

**Output the user sees:**
```
[thinking: 28 tokens]          ← shown collapsed or dimmed
The answer is 391.             ← shown in normal text
```

### 7.4 ThinkingController (aarambh-ai-inference/src/thinking.rs)

```rust
pub struct ThinkingController {
    mode:              ThinkingMode,    // None / Low(256) / Medium(1024) / High(4096)
    in_thinking_block: bool,
    tokens_used:       usize,
    started:           bool,
    closed:            bool,
    pending_force:     Option<ForceToken>,
}

// Called before sampling each generated token:
pub fn take_forced_token(&mut self) -> Option<ForceToken> {
    // queued ThinkEnd wins, otherwise force ThinkStart once when mode != None
}

// Called after every emitted token:
pub fn on_token(
    &mut self,
    token_id: u32,
) -> Option<ForceToken> {
    if self.in_thinking_block {
        if token_id == THINK_END_ID {
            self.in_thinking_block = false;
            return None;
        }
        self.tokens_used += 1;
        if self.tokens_used >= self.mode.budget() {
            return Some(ForceToken::ThinkEnd);   // force close on next step
        }
    }
    None
}
```

`GenerationOutput.text` is the visible answer only. `raw_text` preserves the
actual generated token stream including `<think>` and `</think>`, while
`thinking_text` and `answer_text` are separated for CLI rendering and future
safety/fine-tuning stages. `max_new_tokens` counts every emitted token including
forced thinking markers.

### 7.5 Training the Thinking Engine

The model learns to use `<think>` tokens through two fine-tuning stages:

**Stage 1 — Thinking SFT:** Fine-tune on examples with `(problem, thinking, answer)`:
```
<|user|>
What is 23 × 47?
<|assistant|>
<think>
23 × 47 = 23 × 40 + 23 × 7
23 × 40 = 920
23 × 7  = 161
920 + 161 = 1081
</think>
The answer is 1081.
```

**Stage 2 — GRPO RL:** Generate G=8 completions per math problem. Score by
correctness. The model learns to allocate the right amount of thinking:
- correct + concise thinking → high reward
- wrong answer → negative reward
- excessive empty thinking → lower reward (penalised by format verifier)

---

## 8. KV Cache & Inference

### 8.1 KV Cache

During training, the full sequence is processed in parallel. During inference,
tokens are generated one at a time. Without caching, K and V for all previous
tokens would be recomputed on every step — O(n²) total cost.

The KV cache stores K and V from all previous steps:
```
Step 1: compute K[0], V[0] → cache → output token 1
Step 2: compute K[1], V[1] → cache → attend to K[0..1], V[0..1] → output token 2
Step N: compute K[N], V[N] → cache → attend to K[0..N], V[0..N] → output token N+1
```

Each layer has its own cache. Memory:
```
2 × N_layers × N_kv_heads × max_seq_len × d_head × sizeof(dtype)

Tiny at F32: 2 × 8 × 2 × 512 × 64 × 4 = ~4 MB per sequence
```

Phase 6 implements this in `aarambh-ai-inference`:
- `KvCache` owns one `aarambh-ai-nn::KVCache` per transformer layer and exposes
  mutable layer slices to `AarambhModel::forward_with_cache()`
- `InferenceEngine::from_paths(...)` loads a SafeTensors checkpoint, validates
  tokenizer special IDs, and adjusts the model vocab size to the tokenizer
- `generate_with_callback(...)` prefills the prompt once, decodes one token at a
  time with cache offsets, emits `GenerationStep` values, and stops at
  `<|endoftext|>`, max tokens, or context limit
- `ThinkingMode` forces `<think>` once, budget-tracks thinking content, and
  force-closes with `</think>` when the active budget is reached

### 8.2 Sampling Strategies

After the forward pass gives `logits ∈ ℝ^vocab_size`:

**Greedy:** Always take the highest probability token. Deterministic, fast, but repetitive.

**Temperature:** Scale logits before softmax. τ < 1 = more confident, τ > 1 = more random.
```
P(v) = softmax(logits / τ)[v]
```

**Top-K:** Keep only the K highest-logit tokens, set rest to −∞, then sample.

**Top-P (nucleus):** Keep the smallest set of tokens whose cumulative probability ≥ p.
More dynamic than Top-K — uses more candidates when the model is uncertain.

**Recommended defaults:**
```
Validation (deterministic):  Greedy
Interactive inference:        temperature=0.7, top_p=0.9
Thinking mode Low:            temperature=0.5  (focused, but not collapsed)
Thinking mode Medium:         temperature=0.7, top_p=0.9
Thinking mode High:           temperature=0.8, top_p=0.95  (exploratory)
```

> **Note on Low temperature:** Using temperature=0.3 for Low thinking can cause
> repetitive reasoning loops ("1+1=2, 1+1=2, 1+1=2..."). Temperature=0.5 keeps
> focus while avoiding degenerate repetition. Adjust based on your observations.

### 8.3 Next-Token Prediction View

When you run `aarambh-ai infer --predict-view`, after each token the CLI shows:
```
Input:  "The capital of India is"
────────────────────────────────────────────────────────
Generated so far: "The capital of India is New"

  Next token predictions:
  ══════════════════════════════════════════════════════
  ████████████████████████  48.2%  " Delhi"    ✓ chosen
  █████████████             24.7%  " new"
  █████                      9.1%  " Bombay"
  ██                         4.3%  " the"
  █                          2.1%  " Mumbai"
  ══════════════════════════════════════════════════════
  Temperature: 0.7  |  Top-P: 0.9  |  Step: 2 / 50
────────────────────────────────────────────────────────
```

This is implemented in `aarambh-ai/src/ui/predict_view.rs` from the
`GenerationStep.candidates` returned by the sampler. It is the most powerful
tool for understanding and debugging your model — you can see exactly which
tokens the model is considering and with what confidence.

---

## 9. Training Pipeline

### 9.1 Data Preparation

```
Raw text file
      │
      ▼  tokenise all documents
All token IDs concatenated with <|endoftext|> between docs
      │
      ▼  slide a window of (max_seq_len + 1) tokens
[tokens 0..512]   [tokens 512..1024]   [tokens 1024..1536]   ...
      │
      ▼  split into (input, label) pairs — shift by 1
input:  tokens[0..511]     ← the model sees these
label:  tokens[1..512]     ← what the model must predict

      ▼  batch: stack N windows into one tensor
input_ids:  [batch, seq_len]   dtype: u32
labels:     [batch, seq_len]   dtype: u32
```

### 9.2 Forward Pass

```
token_ids: [batch, seq_len]          ← u32 integer IDs
     │
     ▼ embedding table lookup
[batch, seq_len, d_model]            ← float vectors

     │ repeat × N_layers
     ├── RMSNorm → GQA (+ RoPE, causal mask) → residual add
     └── RMSNorm → SwiGLU FFN → residual add

     │ final RMSNorm
     ▼ LM head (linear)
[batch, seq_len, vocab_size]         ← logits (raw scores)
```

Training uses `AarambhModel::forward_train()`. This path deliberately calls
Candle autograd-compatible RMSNorm and attention operations. The Phase 4 CPU
SIMD and parallel-attention kernels remain inference dispatch paths until custom
backward kernels are implemented.

Token embeddings are initialized with `N(0, 0.02)`. This matters because the
Tiny config ties embeddings to the LM head; unit-scale embeddings would produce
oversized logits and random-model losses around `80` instead of `ln(vocab)`.

### 9.3 Cross-Entropy Loss

For each position in the sequence:
```
loss[b, t] = −log( softmax(logits[b, t, :])[labels[b, t]] )
```

Final loss = `mean(loss)` across all non-padding positions.

**Perplexity** = `exp(loss)`. Lower is better:
- Random model over 32K tokens: PPL = 32,000
- Well-trained Tiny on Shakespeare: PPL < 15
- A useful instruction-following model: PPL < 5 on its domain

### 9.4 Backward Pass & Optimiser (AdamW)

Candle's autograd computes gradients automatically after `loss.backward()`.
The trainer collects each returned `GradStore` into a named gradient map keyed
by the model `VarMap` parameter names, then applies AdamW to the mutable Vars.

AdamW update per parameter (with corrected defaults):
```
m = β₁·m + (1−β₁)·grad               ← gradient direction (EMA)
v = β₂·v + (1−β₂)·grad²              ← gradient variance (EMA)

m̂ = m / (1 − β₁ᵗ)                    ← bias correction
v̂ = v / (1 − β₂ᵗ)

W = W − lr · m̂/(√v̂ + ε) − lr · λ · W  ← update + weight decay
```

**Defaults (match LLaMA 2/3 training):**
```
β₁  = 0.9
β₂  = 0.95    ← NOT 0.999; 0.95 is correct for LLM pretraining
ε   = 1e-8
λ   = 0.1     ← weight decay
```

Weight decay applied only to linear projection weights — NOT to embeddings,
biases, or RMSNorm scale γ.

### 9.5 Gradient Clipping & Accumulation

**Gradient clipping** (before optimiser step):
```
global_norm = sqrt( Σ ‖grad_i‖² )
if global_norm > 1.0:
    for each param: grad *= 1.0 / global_norm
```

**Gradient accumulation** (for small GPUs / CPU):
```
effective_batch = batch_size × grad_accum_steps

for each micro_batch (batch_size=2):
    loss = forward(micro_batch) / grad_accum_steps
    loss.backward()   ← returns this micro-batch's GradStore
    add GradStore into named gradient accumulator
    if step % grad_accum_steps == 0:
        clip_gradients()
        optimiser.step()
        clear accumulator
```

With `grad_accum_steps=16` and `batch_size=2`: effective batch = 32.

### 9.6 Learning Rate Schedule

```
Steps 0 → warmup_steps:
  lr = max_lr × (step / warmup_steps)    ← linear warmup

Steps warmup_steps → max_steps:
  progress = (step − warmup_steps) / (max_steps − warmup_steps)
  lr = min_lr + 0.5 × (max_lr − min_lr) × (1 + cos(π × progress))
  
  where min_lr = max_lr / 10
```

### 9.7 Training Output

```
$ cargo run --release -- train --config configs/tiny_shakespeare.toml

step=1 loss=9.0304 ppl=8352.87 lr=0.000250 grad_norm=0.7182
step=10 loss=9.0241 ppl=8300.43 lr=0.000800 grad_norm=0.7221
eval step=500 val_loss=3.2110 val_ppl=24.80
step=1000 loss=2.8740 ppl=17.71 lr=0.000287 grad_norm=0.9123
```

For a quick CPU validation, use:
```
$ cargo run --release -- train --config configs/tiny_shakespeare_smoke.toml
```

Checkpoint directories contain:
```
model.safetensors       ← model VarMap tensors
optimizer.safetensors   ← AdamW m/v moment tensors
train_state.json        ← step, epoch, micro-step, train/val/best losses
```

`latest.json` and `best.json` point to the active checkpoint directories.

---

## 10. Custom Kernels (aarambh-ai-kernel)

### 10.1 Why a Kernel Crate

The kernel crate is the **only** place in the workspace that contains CUDA C code,
unsafe Rust, and raw pointer arithmetic. All other crates stay 100% safe Rust.
This boundary is intentional: you can verify every layer's correctness in pure Rust
first, then drop in the kernel for speed without touching higher-level code.

### 10.2 Toolchain Requirements for SIMD

The CPU SIMD kernel (`cpu/simd_norm.rs`) uses stable `std::arch` intrinsics with
cached runtime dispatch. No nightly override is required. On x86/x86_64, the
kernel prefers AVX2/FMA by default on this class of CPU, supports AVX512 when
forced with `AARAMBH_SIMD_FORCE=avx512`, and falls back to AVX2 or scalar. Non-x86
targets use the scalar path.

### 10.3 Flash Attention v2

Standard attention materialises the full `[seq, seq]` score matrix. For seq=2048,
16 heads, F32: that is 256 MB allocated and discarded on every single forward pass.
For seq=4096 this is 1 GB.

**Flash Attention** computes the same result without ever storing the full matrix.
It processes K and V in tiles that fit in GPU L2/SRAM, using the online-softmax
trick to maintain a running max and denominator:

```
For each tile of Q (size B_r):
  For each tile of K, V (size B_c):
    S_tile = Q_tile @ K_tileᵀ / sqrt(d_head)    ← small tile only
    m_new  = max(m_old, row_max(S_tile))
    l_new  = e^(m_old − m_new) · l_old + row_sum(e^(S_tile − m_new))
    O      = (l_old / l_new) · e^(m_old − m_new) · O + e^(S_tile − m_new) @ V_tile
```

Memory: O(n) instead of O(n²). For n=4096: ~16× less HBM access on GPU.

**Integration with aarambh-ai-nn:**
```rust
// attention.rs — dispatch based on device at runtime
let output = aarambh_ai_kernel::dispatch::attention_forward(&q, &k, &v, mask, scale)?;
```

On CPU F32 tensors, Phase 4 uses the Rayon attention kernel. CUDA execution stays
as a stub until Phase 14, so CUDA devices keep using the Candle fallback.

### 10.4 Fused Kernels (GPU)

Each fused kernel eliminates one or more intermediate tensor allocations:

| Kernel | What it eliminates | Speedup |
|---|---|---|
| `rms_norm_fused.cu` | Temp buffer between RMS compute + normalise passes | ~2.8× |
| `rope_apply.cu` | Two separate Q and K element-wise ops become one kernel | ~1.5× |
| `swiglu_fused.cu` | Intermediate `gate` and `up` tensors written to HBM | ~2.0× |

### 10.5 CPU Kernels (for your i3)

Even without CUDA, the kernel crate gives speedups on CPU:

- **SIMD RMSNorm** (`cpu/simd_norm.rs`): Cached AVX2/FMA, AVX512, AVX2, and scalar dispatch for F32 CPU tensors.
- **Parallel attention** (`cpu/parallel_attn.rs`): Uses `rayon` to run all
  attention heads in parallel across all 4 logical cores. Tiny model has 6 heads
  → near-linear speedup. Uses stable Rust only.

### 10.6 Build System

`build.rs` detects NVCC at build time. If not found, CUDA stubs are skipped — the
crate still compiles and the dispatch layer falls back to Candle where needed:

```rust
// build.rs
fn main() {
    println!("cargo:rustc-check-cfg=cfg(aarambh_cuda_stubs)");
    if which::which("nvcc").is_ok() {
        cc::Build::new()
            .cuda(true)
            .file("kernels/flash_attention.cu")
            .file("kernels/flash_attn_bwd.cu")
            .file("kernels/rms_norm_fused.cu")
            .file("kernels/rope_apply.cu")
            .file("kernels/swiglu_fused.cu")
            .compile("aarambh_cuda_stubs");
        println!("cargo:rustc-cfg=aarambh_cuda_stubs");
    } else {
        println!("cargo:warning=nvcc not found; CUDA kernel stubs are disabled");
    }
}
```

CPU-only machines always build cleanly. On CUDA machines, Phase 4 validates the
CUDA build plumbing; real GPU kernels are implemented later.

---

## 11. Quantisation (aarambh-ai-quant)

### 11.1 Why Quantise

The Tiny model at F32 is ~100 MB. At INT4, it drops to ~13 MB. More importantly:
CPU inference is **memory-bandwidth bound**, not compute-bound. INT4 weights need
8× less bandwidth from RAM → ~4–6× faster inference on your i3 without any GPU.

### 11.2 Absmax INT8 (start here)

Simplest scheme — one scale per tensor:
```
scale   = max(|W|) / 127
W_int8  = round(W / scale)          ← store as i8
W_float = W_int8 × scale            ← dequantise before compute
```
4× compression. Perplexity increase: < 1%. Implement this first, easiest to debug.

### 11.3 GPTQ INT4

GPTQ processes weights column by column, using the Hessian `H = 2XᵀX`
(computed from 128 calibration samples) to correct quantisation error.

**Important:** GPTQ requires the **inverse** of H, computed via Cholesky decomposition
for numerical stability. Do not invert naively (matrix inverse of large H is unstable):

```
H = 2 × Xᵀ × X                          ← compute from calibration activations
H_chol = cholesky(H + δI)                ← δ = 1e-6 damping for stability
H_inv = solve via back-substitution       ← use Cholesky factorisation

For each column j:
  1. Quantise W[:,j] to 4-bit → W_q[:,j]
  2. error = W[:,j] − dequant(W_q[:,j])
  3. Distribute error to remaining columns:
     W[:,j+1:] -= error × H_inv[j,j]⁻¹ × H_inv[j, j+1:]
```

8× compression. Perplexity increase: 2–4%. Imperceptible in practice.

**Simpler alternative for early implementation:** Use the diagonal Hessian approximation
`H_inv ≈ diag(H)⁻¹` — much faster and easier to implement, slightly lower quality.
Phase 8 uses the full damped Cholesky path for the public `cholesky_invert()`
API and returns an error if the matrix is still not positive definite after
damping retries.

### 11.4 AWQ INT4

Identifies the ~1% of weight channels that matter most (using activation
statistics), scales them up before quantisation, then scales back:
```
s_j = mean_X(|X[:,j]|)^α         ← per-channel scale from calibration
W_awq = quant(W × diag(s)) × diag(s)⁻¹
```
No Hessian inversion needed → faster calibration than GPTQ, similar quality.
Implement this before GPTQ if you want INT4 sooner.

### 11.5 Quantisation-Aware Training (qat.rs)

QAT inserts **fake quantisation nodes** into the model during fine-tuning.
Forward pass: simulate INT4 quantisation error. Backward pass: straight-through estimator.
This teaches the model to be robust to quantisation before the weights are actually quantised.

```
# QAT fine-tune → then export to INT4
aarambh-ai finetune sft --qat --bits 4 --base checkpoint.safetensors
aarambh-ai quantise --method gptq --bits 4 --input qat_checkpoint.safetensors
```

QAT typically recovers 1–2 PPL points vs post-training quantisation alone.

### 11.6 GGUF Q4_K_M

Block-wise quantisation for Aarambh GGUF artifacts. Q4_K_M stores 256 weights
per block with a scale/min pair:
```
Block layout: [scale_f16: 2 bytes][min_f16: 2 bytes][weights: 128 bytes]
              = 132 bytes for 256 weights
```

Phase 8 writes an Aarambh GGUF container with `GGUF` magic, model metadata, and
raw quantized tensor payloads. `load_gguf()` dequantises tensors back into an
`AarambhModel` for compatibility with the existing Candle inference engine, while
the on-disk model remains compact.

### 11.7 KV Cache Quantisation

K and V can be stored as INT8 in the cache, dequantised to F32 only when needed
for the attention dot product. Halves KV cache memory with no perceptible quality loss.

Phase 8 provides `QuantisedKvCache`, which stores each layer's K/V tensors as
INT8 absmax tensors between decode steps and returns F32 tensors to the
attention kernel.

### 11.8 Weight Conversion (convert.rs)

`aarambh-ai-weights/src/convert.rs` loads HuggingFace-format checkpoints
(LLaMA 2/3, Mistral, Qwen, etc.) and converts them to aarambh-ai's key naming
and tensor layout:

- Renames keys: `model.layers.0.self_attn.q_proj.weight` → `blocks.0.attn.wq`
- **Pragmatic GQA handling:** If the source has more KV heads, we take the first `cfg.n_kv_heads` via strict slicing. If the source has fewer, we panic (unsupported). No complex redistributive reshaping.
- Handles tied vs untied LM head (some HF models store them separately)
- Converts dtype as needed (F16 → F32, BF16 → F32)
- Reads both `model.safetensors.index.json` sharded checkpoints and single
  `.safetensors` files.

```bash
aarambh-ai convert \
  --input  /path/to/hf_llama2_7b/ \
  --output checkpoints/llama2_7b_aarambh.safetensors \
  --arch   llama2
```

### 11.9 Model Sizes After Quantisation

| Scale | F32 | BF16 | INT8 | INT4 (GPTQ) | GGUF Q4_K_M |
|---|---|---|---|---|---|
| Tiny   | 100 MB  | 50 MB  | 25 MB  | 13 MB   | 13 MB  |
| Small  | 450 MB  | 225 MB | 113 MB | 57 MB   | 59 MB  |
| Medium | 1.37 GB | 685 MB | 342 MB | 171 MB  | 179 MB |
| Large  | 4.9 GB  | 2.4 GB | 1.2 GB | 620 MB  | 645 MB |

Tiny-Q4 in 13 MB: embeddable in other Rust apps, WASM-capable (with limitations).

---

## 12. Fine-Tuning (aarambh-ai-finetune)

### 12.1 LoRA — Low-Rank Adaptation

Full fine-tuning updates all 25M+ parameters. LoRA constrains updates to a
low-rank matrix product — training orders of magnitude fewer parameters.

For a frozen pre-trained weight W₀ ∈ ℝ^(d×k):
```
ΔW = B · A      where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r ≪ min(d,k)

forward:  h = W₀x + (α/r) · BAx    ← α=r typically, scale = 1.0
```

During training: W₀ is **frozen** (no gradients). Only A and B are trained.

**Parameter reduction (Small model, rank=16, targeting Q/K/V/O):**
```
Full fine-tune:  117M parameters with gradients → ~2.5 GB memory
LoRA fine-tune:  4 × 2 × 768 × 16 = 98,304 parameters → <0.1% of total
```

**At inference:** Merge LoRA into base → zero latency overhead:
```rust
W_merged = W₀ + (alpha / rank) * (lora_b @ lora_a)
```

### 12.2 QLoRA — Fine-Tuning on Your i3

QLoRA combines INT4 base model + BF16 LoRA adapters.

**Critical implementation detail:** On every forward pass, the INT4 base weights
must be dequantised to compute gradients through the LoRA branch. `qlora.rs`
calls `dequant::dequantise_i4()` inside `forward()` — this enables autograd to
flow through the dequantised values into the LoRA parameters:

```rust
// qlora.rs forward — simplified
fn forward(&self, x: &Tensor) -> Result<Tensor> {
    // Dequant on-the-fly (not stored) to enable autograd
    let w_bf16 = dequant::dequantise_i4(&self.base_int4, &self.scales)?;
    let base_out = x.matmul(&w_bf16.t()?)?;
    let lora_out = x.matmul(&self.lora_a.t()?)?.matmul(&self.lora_b.t()?)?;
    Ok((base_out + (lora_out * self.scale)?)?)
}
```

Memory comparison for Small model fine-tuning:
```
Full fine-tune:  ~2.5 GB   ← OOM on 8 GB i3
LoRA fine-tune:  ~0.9 GB   ← fits, but tight
QLoRA fine-tune: ~0.4 GB   ← comfortable on 8 GB
```

**QLoRA lets you fine-tune the 117M Small model on your i3 laptop.**

### 12.3 SFT — Supervised Fine-Tuning with Loss Masking

For instruction following, you only want loss on the **assistant's response**,
not on the system prompt or user message:

```
<|user|>\nWhat is 5 + 3?\n<|assistant|>\nThe answer is 8.<|endoftext|>
─────────────────────────────────────────────────────────────────────
loss mask: 0  0  0  0  0  0  0  0  0    1  1  1  1  1  1  1  1  1
           ↑ user tokens (no gradient)  ↑ assistant tokens (gradient here)
```

`SftTrainer` finds the `<|assistant|>` position and masks all prior positions.

**Thinking SFT format** (teaches the model to use `<think>`):
```json
{"instruction": "What is 23 × 47?",
 "thinking":    "23 × 47 = 23 × 40 + 23 × 7 = 920 + 161 = 1081",
 "response":    "The answer is 1081."}
```

Both the thinking block and the response are included in the loss mask.

### 12.4 GRPO — Teaching the Model to Think Better

GRPO (introduced in DeepSeek-R1, 2025) improves reasoning without a separate reward model.

**IMPORTANT:** GRPO must be used with a **deterministic verifier** (Math, Code, or Format).
Self-Critique is too noisy for GRPO advantages; it is used exclusively for the Replay Buffer
in Phase 12.

```
1. Sample G=8 completions for each prompt using the current LoRA policy (temperature=0.8)
2. Score each with deterministic verifier(completion, ground_truth) → score ∈ [0, 1]
3. Normalise scores within the group:
   advantage_i = (score_i − mean(scores)) / (std(scores) + 1e-8)
4. Replay prompt + completion through the training graph:
   policy_log_probs = LoraAarambhModel::forward_train(...)
   ref_log_probs    = frozen AarambhModel::forward_train(...)
5. GRPO loss:
   L = −mean(policy_log_prob(completion_i) × advantage_i) + β × KL(current ‖ reference)
6. Backward + AdamW step on LoRA parameters only
```

> **Naming note:** The loss function parameters are `policy_log_probs` (from the
> model being trained) and `ref_log_probs` (from the frozen reference checkpoint).
> Keep these names explicit in `grpo.rs` — they refer to different models.

Sampling is graph-free and stores only completion token IDs/text. Differentiable
log-probs are recomputed by replay so GRPO never backpropagates through the
cached inference path. The KL penalty keeps the model close to the SFT
checkpoint so it doesn't drift.

**Built-in verifiers:**
- `MathVerifier` — parse final number from response, compare to ground truth
- `FormatVerifier` — reward responses with a proper `<think>...</think>` block
- `CodeVerifier` — compile and run code snippet, check output (future)
- `CompositeVerifier` — weighted combination of multiple verifiers

---

## 13. Safety Layer (aarambh-ai-safety)

### 13.1 Architecture

The `SafetyGuard` wraps the mutable `InferenceEngine` and intercepts every
generation call. Live streaming callbacks are buffered while safety is enabled,
so output guardrails run before generated text is printed.

```
User prompt
     │
     ▼
┌────────────────────────────────┐
│  Input Guardrails              │
│  1. Prompt injection check     │
│  2. Jailbreak detection        │
│  3. PII detection              │
│  → Allow / Redact / Block      │
└──────────────┬─────────────────┘
               │ (Allow or Redacted prompt)
               ▼
┌────────────────────────────────┐
│  InferenceEngine.generate()    │
│  (with ThinkingController)     │
└──────────────┬─────────────────┘
               │ raw response
               ▼
┌────────────────────────────────┐
│  Output Guardrails             │
│  1. Toxicity scoring           │
│  2. PII redaction              │
│  3. Format validation          │
│  → Allow / Redact / Block /    │
│     Regenerate(max 3 tries)    │
└──────────────┬─────────────────┘
               │
               ▼
     Response to user
     + SafetyEvent logged
```

### 13.2 Input Guardrails

**Prompt Injection (`input/injection.rs`):**
Detects attempts to override system instructions:
- Pattern library: `"ignore previous instructions"`, `"disregard your"`,
  `"new system prompt:"`, instruction-like XML/JSON in user input
- Structural anomaly scoring: many newlines, role-switching phrases
- Returns `InjectionScore { score: f32, patterns: Vec<String> }`

**Jailbreak Detection (`input/jailbreak.rs`):**
- Role-play bypasses: `"pretend you are"`, `"act as if"`, `"you are DAN"`
- Character encoding tricks: Base64, Leetspeak, Unicode lookalikes
- Known jailbreak template similarity

**PII Detection (`input/pii.rs`):**
- Email, phone (international), SSN/national ID, credit cards (Luhn check),
  API keys (high-entropy string heuristics: `sk-`, `ghp_`, long hex)
- Action per PII: Block / Redact / Warn (configured in SafetyPolicy)

### 13.3 Output Guardrails

**Toxicity Scoring (`output/toxicity.rs`):**
Five categories: hate speech, violence, sexual content, self-harm, illegal activity.
Returns `ToxicityScore { overall: f32, categories: HashMap<Category, f32> }`.
Action above threshold (default 0.7): Block or Regenerate (up to 3 tries).

**PII Redaction (`output/pii_redact.rs`):**
Same patterns as input side — if model accidentally outputs a phone number or email,
it is replaced with `[REDACTED_PHONE]` / `[REDACTED_EMAIL]`.

### 13.4 Safety Policy

```rust
pub struct SafetyPolicy {
    // Input
    pub check_prompt_injection: bool,
    pub injection_threshold:    f32,
    pub check_jailbreak:        bool,
    pub jailbreak_threshold:    f32,
    pub input_pii:              PiiPolicy,    // Off/Warn/Redact/Block
    pub max_prompt_chars:       Option<usize>,

    // Output
    pub check_toxicity:         bool,
    pub toxicity_threshold:     f32,          // 0.0–1.0, default 0.7
    pub output_pii:             PiiPolicy,

    // On violation
    pub on_input_violation:     ViolationAction,  // Allow/Warn/Block
    pub on_output_violation:    ViolationAction,  // Block/Regenerate/Warn
    pub max_regenerations:      usize,
    pub audit_enabled:          bool,
    pub audit_path:             Option<PathBuf>,
}

impl SafetyPolicy {
    pub fn strict()     -> Self { /* all checks on, low thresholds  */ }
    pub fn permissive() -> Self { /* injection + jailbreak only     */ }
    pub fn research()   -> Self { /* log only, nothing blocked      */ }
}
```

### 13.5 Audit Logging

Every safety event → `safety_audit.jsonl`:
```json
{
  "timestamp_unix_ms": 1719000000000,
  "prompt_hash":     "a3f5bc...",   ← SHA-256 of prompt (never the text)
  "stage":           "input",
  "verdict":         "block",
  "triggered_rules": ["input.ignore_previous_instructions"],
  "latency_ms":      2
}
```

The prompt text is **never** logged — only its hash. Enables audit trails without storing user data.

---

## 14. Self-Learning Loop (aarambh-ai-selflearn)

### 14.1 Overview & Design

> **Roadmap placement:** This is **Phase 12** — after Safety (Phase 11) and before
> GPU Scale-Up (Phase 13). Self-learning is a core v1.0 feature, not a post-release
> addon. Every checkpoint released in Phase 15 supports `--self-learn` out of the box.

After pretraining and SFT, the model knows how to follow instructions. The
self-learning loop lets it **keep getting better after deployment** — with no
human labels required. It does this by generating its own training signal
from its own outputs.

**Critical Design Distinction:**
- **Online GRPO** uses a **deterministic verifier** (Math/Code) to compute advantages. Self-Critique is never used for GRPO.
- **Self-Critique** is a stateless free function that scores the *best* answer from GRPO. Its score is used **only** to decide whether to store the answer in the Replay Buffer.

Three mechanisms work together as one loop:

```
┌─────────────────────────────────────────────────────────┐
│                  SELF-LEARNING LOOP                     │
│                                                         │
│  User Prompt                                            │
│       │                                                 │
│       ▼                                                 │
│  ┌─────────────────────────────┐                        │
│  │  Online GRPO                │                        │
│  │  Generate N=4 completions   │                        │
│  │  Score via DETERMINISTIC    │ ← Math/Code verifier   │
│  │  verifier (not SelfCritique)│                        │
│  │  Compute advantages         │                        │
│  │  Mini gradient step         │ ← trains immediately   │
│  └──────────────┬──────────────┘                        │
│                 │ best completion(s)                    │
│                 ▼                                       │
│  ┌─────────────────────────────┐                        │
│  │  Self-Critique (free fn)    │                        │
│  │  Model reads its own output │                        │
│  │  Assigns quality score      │                        │
│  │  Optionally rewrites        │                        │
│  └──────────────┬──────────────┘                        │
│                 │ scored (prompt, response) pair        │
│                 ▼                                       │
│  ┌─────────────────────────────┐                        │
│  │  Replay Buffer              │                        │
│  │  Store if score ≥ threshold │                        │
│  │  Every K steps: replay      │ ← periodic fine-tune  │
│  │  Evict oldest low-quality   │                        │
│  └─────────────────────────────┘                        │
│                                                         │
│  Result: model returned to user                         │
│  Side effect: model slightly better than before         │
└─────────────────────────────────────────────────────────┘
```

**CPU-safe mode (i3):** N=2 completions instead of N=8, no gradient step during
inference (gradient accumulates offline), replay fine-tunes only every 500 steps.
The loop still works — just slower to improve.

**GPU mode (Kaggle):** N=8 completions, immediate gradient step, replay every 50 steps.

### 14.2 Online GRPO

The core mechanism. When the user asks a question, the model generates N answers
**in the same inference call**, scores them using a **deterministic verifier**,
then takes one gradient step before returning the best answer.

```rust
// online_grpo.rs

pub struct OnlineGrpo {
    engine:    InferenceEngine,
    ref_model: AarambhModel,     // frozen reference — prevents policy collapse
    optimizer: AdamW,
    config:    OnlineGrpoConfig,
}

pub struct OnlineGrpoConfig {
    pub n_completions:  usize,   // CPU: 2  |  GPU: 8
    pub temperature:    f32,     // 0.8 for diversity
    pub kl_coeff:       f64,     // 0.01 — keep close to reference
    pub max_grad_norm:  f64,     // 1.0
    pub skip_on_cpu:    bool,    // if true: accumulate gradients, don't step inline
}
```

**The scoring problem:** Online GRPO needs a verifier to score completions.
We use **deterministic verifiers** (MathVerifier, CodeVerifier) when ground truth
exists. For open-ended tasks, we **skip GRPO** and rely purely on the Replay Buffer
(SFT) for improvement. Self-Critique is never used for GRPO advantages.

**Catastrophic forgetting prevention:**
- The frozen `ref_model` provides the KL penalty anchor
- LoRA adapters are used — only rank-16 adapters are updated, not the full model
- Learning rate for online steps: `1e-5` (10× smaller than SFT)

```
Online step memory (Tiny, LoRA rank=16):
  N=2: ~180 MB   ← fits on i3 8 GB comfortably
  N=8: ~450 MB   ← needs GPU or very careful memory management
```

### 14.3 Experience Replay Buffer

Stores the best (prompt, response) pairs the model has generated. Periodically
re-trains on them to consolidate learning without forgetting.

```rust
// replay.rs

pub struct ReplayEntry {
    pub prompt:    String,
    pub response:  String,
    pub score:     f32,          // from self-critique (replay-only)
    pub timestamp: u64,
    pub topic:     String,       // inferred category for diversity sampling
}

pub struct ReplayBuffer {
    entries:      Vec<ReplayEntry>,
    capacity:     usize,         // CPU: 500  |  GPU: 5000
    min_score:    f32,           // only store if score ≥ 0.7
}

impl ReplayBuffer {
    pub fn push(&mut self, entry: ReplayEntry)
      // if full: evict lowest-scoring entry
      // never evict if score > 0.9 (high-quality examples are precious)

    pub fn sample_batch(&self, n: usize) -> Vec<&ReplayEntry>
      // prioritised sampling: higher score → more likely to be sampled
      // diversity: at most 2 entries per topic per batch

    pub fn save_jsonl(&self, path: &Path) -> Result<()>
    pub fn load_jsonl(path: &Path) -> Result<Self>
      // persists across restarts — replay buffer survives process restart
}
```

**Replay fine-tune trigger:**

```
CPU (i3):   every 500 online steps → 1 epoch over 32 sampled entries
GPU:        every  50 online steps → 1 epoch over 128 sampled entries
```

This is the mechanism that prevents forgetting. The model continuously sees a
curated sample of its best past work, staying sharp on everything it has learned.

### 14.4 Self-Critique Loop (Replay-Only)

After generating a response (the best candidate from GRPO), the model reads it and
assigns a quality score. This score drives **only the replay buffer**, never GRPO.

**Implementation:** SelfCritique is a **stateless free function** to avoid Rust
borrow-checker issues when borrowing the InferenceEngine mutably.

```rust
// critique.rs — free function, not a struct

pub fn critique_response(
    engine: &mut InferenceEngine,
    prompt: &str,
    response: &str,
    config: &CritiqueConfig,
) -> Result<(String, f32)> {
    // Builds critique prompt template
    // Calls engine.generate() for ~50 tokens
    // Parses JSON: {"score": 0.85, "reason": "..."}
    // Fallback: if JSON malformed → score = 0.5
    // If score < rewrite_threshold: re-generate at temperature=0.5, score again
    // After max_rewrites: return best version seen
}
```

**Critique prompt template:**

```
<|user|>
Rate the following response on a scale of 0.0 to 1.0.
Consider: accuracy, clarity, completeness, reasoning quality.

Question: {original_prompt}
Response: {model_response}

Reply with ONLY a JSON object: {"score": <float>, "reason": "<one sentence>"}
<|assistant|>
```

The model parses the JSON, extracts the score, and optionally rewrites:

```
score ≥ 0.85  →  Accept as-is, store in replay buffer
score ≥ 0.70  →  Accept, store with lower priority
score < 0.70  →  Rewrite (re-generate with temperature=0.5, critique again)
score < 0.50  →  Discard, do not store
```

**Why self-critique works (for replay):** The model in "critic" mode has access
to its own weights and training. Even a noisy critic signal is enough to distinguish
clearly good from clearly bad responses — which is all the replay buffer needs.

**CPU optimisation:** On the i3, the critique is a short inference call (~50 tokens).
Total overhead per user turn: ~200ms extra. Acceptable.

### 14.5 CPU-safe Mode (i3)

The full loop runs on your i3 laptop with these constraints:

| Setting | CPU (i3) | GPU (Kaggle) |
|---|---|---|
| N completions | 2 | 8 |
| Gradient step | Deferred (accumulate offline) | Inline (immediate) |
| Replay trigger | Every 500 steps | Every 50 steps |
| Replay batch size | 32 | 128 |
| Replay buffer cap | 500 entries | 5,000 entries |
| Self-critique rewrites | Max 1 | Max 3 |
| Extra memory per turn | ~180 MB | ~450 MB |
| LoRA rank | 8 | 16 |

**Deferred gradient step (CPU mode):** Gradients accumulate in memory across
multiple turns. Every 500 turns, a single gradient step is taken during a quiet
moment (e.g., idle detection or explicit `--flush-gradients` CLI flag). This
means the model doesn't improve turn-by-turn on the i3, but it does improve
session-by-session.

### 14.6 Full Loop Flow

```rust
// loop.rs — SelfLearnLoop owns OnlineGrpo and ReplayBuffer.
// SelfCritique is a free function, not a field.

pub struct SelfLearnLoop {
    pub online_grpo: OnlineGrpo,   // owns InferenceEngine
    pub replay: ReplayBuffer,
    pub config: SelfLearnConfig,
}

impl SelfLearnLoop {
    pub fn generate_and_learn(
        &mut self,
        prompt: &str,
        generate_cfg: &GenerateConfig,
        verifier: &dyn Verifier,     // deterministic verifier for GRPO
    ) -> Result<SelfLearnResponse> {

        // 1. Safety check input (applied at binary level)
        // 2. Online GRPO: generate N completions, score with deterministic verifier, mini-step
        let (best_response, policy_log_probs) =
            self.online_grpo.generate_and_step(prompt, generate_cfg, verifier)?;

        // 3. Self-critique: free function borrows engine, scores and optionally rewrites
        let (final_response, score) = critique_response(
            &mut self.online_grpo.engine(),  // explicit borrow
            prompt,
            &best_response,
            &self.config.critique,
        )?;

        // 4. Store in replay buffer if score is good enough
        if score >= self.config.replay.min_score {
            self.replay.push(ReplayEntry {
                prompt:    prompt.to_string(),
                response:  final_response.clone(),
                score,
                timestamp: now_unix(),
                topic:     infer_topic(prompt),
            });
        }

        // 5. Periodic replay fine-tune
        if self.replay.should_replay(self.online_grpo.step_count()) {
            self.replay_finetune()?;
        }

        // 6. Track metrics
        self.metrics.record(score, prompt);

        Ok(SelfLearnResponse {
            response: final_response,
            score,
            stored_in_replay: score >= self.config.replay.min_score,
        })
    }
}
```

### 14.7 SelfLearnConfig

```rust
pub struct SelfLearnConfig {
    // Mode
    pub mode: SelfLearnMode,       // Cpu / Gpu / Disabled

    // Online GRPO
    pub n_completions: usize,      // 2 (CPU) or 8 (GPU)
    pub online_lr: f64,            // 1e-5 — very small for online steps
    pub kl_coeff: f64,             // 0.01
    pub lora_rank: usize,          // 8 (CPU) or 16 (GPU)

    // Replay buffer
    pub replay_capacity: usize,    // 500 (CPU) or 5000 (GPU)
    pub replay_min_score: f32,     // 0.7 — minimum quality to store
    pub replay_every_n_steps: usize, // 500 (CPU) or 50 (GPU)
    pub replay_batch_size: usize,  // 32 (CPU) or 128 (GPU)
    pub replay_path: PathBuf,      // persists buffer to disk

    // Self-critique
    pub critique_enabled: bool,    // true
    pub rewrite_threshold: f32,    // 0.7 — rewrite if score < this
    pub max_rewrites: usize,       // 1 (CPU) or 3 (GPU)
}

impl SelfLearnConfig {
    pub fn for_cpu() -> Self  // i3-safe defaults
    pub fn for_gpu() -> Self  // Kaggle defaults
    pub fn disabled() -> Self // standard inference, no self-learning
}
```

**CLI flag:**
```bash
# Enable self-learning in CPU mode
aarambh-ai infer --model checkpoints/tiny_sft.safetensors \
                 --self-learn cpu \
                 --replay-path data/replay_buffer.jsonl \
                 --prompt "What is recursion?"

# Flush accumulated gradients manually (CPU mode)
aarambh-ai selflearn flush-gradients \
                 --model checkpoints/tiny_sft.safetensors \
                 --replay-path data/replay_buffer.jsonl
```

---

## 15. Crate-by-Crate Reference

| Crate | Layer | Key Types | Dependencies |
|---|---|---|---|
| `aarambh-ai-core` | 0 | `ModelConfig`, `TrainConfig` (incl. `eval_steps`), `Device`, `DType`, `AarambhError`, `Result<T>`, `Forward`, `TokenizerLike` | `candle-core`, `serde`, `thiserror` |
| `aarambh-ai-tokenizer` | 1 | `BpeTokenizer impl TokenizerLike`, `Vocab`, special token IDs | `core`, `tokenizers` |
| `aarambh-ai-data` | 1 | `DataLoader`, `TextDataset`, `JsonlDataset`, `Batch` | `core`, `tokenizer` |
| `aarambh-ai-nn` | 2 | `RMSNorm`, `RopeCache`, `GroupedQueryAttention`, `SwiGluFfn`, `TransformerBlock` | `core`, `candle-nn`, `kernel` |
| `aarambh-ai-kernel` | 2 | `flash_attn::forward()`, `fused_norm::rms_norm()`, dispatch | `core`, `candle-core`, `cc`, `rayon` |
| `aarambh-ai-model` | 3 | `AarambhModel`, `TokenEmbedding`, `LmHead` | `core`, `nn` |
| `aarambh-ai-weights` | 3 | `save_model()`, `load_model()`, `GgufReader`, `convert_hf()` (pragmatic slicing) | `core`, `model`, `safetensors` |
| `aarambh-ai-quant` | 3 | `quantise_i8()`, `GptqQuantiser`, `AwqQuantiser`, `QuantisedKvCache`, `QatNode` | `core`, `model` |
| `aarambh-ai-train` | 4 | `Trainer`, `AdamW`, `CosineScheduler`, `CheckpointManager` | `core`, `model`, `data`, `weights` |
| `aarambh-ai-finetune` | 4 | `LoraLayer`, `inject_lora()`, `merge_lora()`, `SftTrainer`, `GrpoTrainer` (deterministic verifier) | `core`, `model`, `train`, `quant` |
| `aarambh-ai-inference` | 5 | `InferenceEngine`, `KvCache`, `Sampler`, `ThinkingController` | `core`, `model`, `weights` |
| `aarambh-ai-safety` | 5 | `SafetyGuard`, `SafetyPolicy`, `SafetyVerdict` | `core`, `inference` |
| `aarambh-ai-selflearn` | 5 | `SelfLearnLoop` (owns OnlineGrpo, Replay), `critique_response` (free fn), `LearningMetrics` | `core`, `inference`, `finetune` |
| `aarambh-ai` (binary) | 6 | CLI commands: train / infer / finetune / quantise / convert / eval / selflearn | all crates |

---

## 16. Data Flow Across the Workspace

```
Raw text
   │
   ▼ aarambh-ai-tokenizer
Token IDs (Vec<u32>)
   │
   ▼ aarambh-ai-data
Batched Tensors (input_ids, labels)
   │
   ▼ aarambh-ai-nn + aarambh-ai-kernel
Transformer computations (per block)
   │
   ▼ aarambh-ai-model
Logits [batch, seq, vocab_size]
   │
   ├──▶ aarambh-ai-train ──▶ AdamW ──▶ updated weights ──▶ aarambh-ai-weights (save)
   │
   ├──▶ aarambh-ai-finetune ──▶ LoRA/SFT/GRPO ──▶ adapter weights (save)
   │
   ├──▶ aarambh-ai-quant ──▶ INT4 weights ──▶ aarambh-ai-weights (GGUF save)
   │
   ├──▶ aarambh-ai-inference ──▶ aarambh-ai-safety ──▶ safe response to user
   │
   └──▶ aarambh-ai-selflearn
           │
           ├── OnlineGrpo: N completions → deterministic verifier → score → mini AdamW step (LoRA only)
           ├── SelfCritique (free fn): model scores own output → rewrite if low quality (replay-only)
           ├── ReplayBuffer: store good pairs → periodic SFT replay
           └── response returned to user (model is now slightly smarter)

External HF checkpoint
   │
   ▼ aarambh-ai-weights (convert.rs) — pragmatic slicing
aarambh-ai SafeTensors checkpoint
   │
   └──▶ aarambh-ai-inference / aarambh-ai-finetune / aarambh-ai-quant / aarambh-ai-selflearn
```

---

## 17. Memory & Compute Estimates

### Training Memory (BF16, per scale)

| Scale | Weights | Gradients | AdamW States | Activations | Total |
|---|---|---|---|---|---|
| Tiny   | 50 MB   | 50 MB   | 200 MB  | ~100 MB | ~0.4 GB |
| Small  | 225 MB  | 225 MB  | 900 MB  | ~450 MB | ~1.8 GB |
| Medium | 685 MB  | 685 MB  | 2.7 GB  | ~1.4 GB | ~5.5 GB |
| Large  | 2.4 GB  | 2.4 GB  | 9.7 GB  | ~3.8 GB | ~18 GB  |

> AdamW states = 4× weights (m and v maintained in F32 even during BF16 training).

### CPU Inference Memory (F32 weights + KV cache at max context length)

| Scale | Weights | KV Cache | Total |
|---|---|---|---|
| Tiny   | 100 MB  | 4 MB    | ~104 MB  |
| Small  | 450 MB  | 24 MB   | ~474 MB  |
| Medium | 1.37 GB | 192 MB  | ~1.56 GB |
| Large  | 4.9 GB  | 384 MB  | ~5.27 GB |

### CPU Inference Memory (INT4 GGUF weights + KV cache at max context length)

| Scale | Weights (Q4_K_M) | KV Cache | Total |
|---|---|---|---|
| Tiny   | 13 MB   | 4 MB    | ~17 MB   |
| Small  | 59 MB   | 24 MB   | ~83 MB   |
| Medium | 179 MB  | 192 MB  | ~371 MB  |
| Large  | 645 MB  | 384 MB  | ~1.03 GB |

---

## 18. Hardware Strategy

### Your Local Machine (i3-1115G4, 8 GB RAM, Pop OS)

**Use exclusively for Tiny scale.** Everything works:
- Full training loop on tiny_shakespeare.txt (~1 MB, public domain)
- All unit and integration tests
- Inference with predict-view
- Tokeniser training
- Checkpoint save/load
- QLoRA fine-tuning of Small (400 MB peak)
- INT4 inference of Medium (283 MB with Q4_K_M)

**Recommended Tiny training config:**
```toml
batch_size        = 2
grad_accum_steps  = 16       # effective batch = 32
max_seq_len       = 256      # shorter than model max, saves RAM
dataset           = "data/tiny_shakespeare.txt"
learning_rate     = 1e-3
max_steps         = 5000
warmup_steps      = 200
device            = "cpu"
dtype             = "f32"
beta2             = 0.95     # explicit — do not use default 0.999
eval_steps        = 500
```

Expected: ~2 hours for 5K steps. PPL should drop below 15 on Shakespeare.
The predict-view will show coherent English candidates after ~2K steps.

### Kaggle GPU

| Scale | GPU | Dtype | Batch | Expected Speed |
|---|---|---|---|---|
| Small  | T4 16 GB   | BF16 | 16  | ~800 tok/s  |
| Medium | P100 16 GB | BF16 | 8   | ~250 tok/s  |
| Large  | A100 40 GB | BF16 | 16  | ~380 tok/s  |

Switch `device = "cuda"`, `dtype = "bf16"` in the config. Zero code changes.
Download checkpoints from Kaggle output → run inference locally.

### Self-Learning Overhead on i3

With `--self-learn cpu` enabled on Tiny model:

| Step | Extra time per turn | Extra memory |
|---|---|---|
| Generate N=2 completions | +1.5× inference time | +50 MB |
| Self-critique (50 tokens) | +0.3× inference time | negligible |
| Gradient accumulation | +10ms per turn | +150 MB (LoRA grads) |
| Replay fine-tune (every 500) | ~2 min batch, runs async | +200 MB during replay |

Total extra memory on i3 with self-learn: ~**400 MB peak**. Stays well under 8 GB with Tiny.

| Kernel | vs. candle baseline |
|---|---|
| Flash Attention v2 | ~3.5× |
| Fused RMSNorm      | ~2.8× |
| Fused RoPE         | ~1.5× |
| Fused SwiGLU       | ~2.0× |
| End-to-end Tiny    | ~2.8× |
