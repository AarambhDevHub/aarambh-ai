# Changelog

## [0.8.0] - 2026-06-28

### Added

- **`aarambh-ai-quant` crate implementation (Phase 8)**
  - INT8 absmax quantisation and dequantisation
  - Packed INT4 affine quantisation with per-group scales/zero-points
  - AWQ activation-scale computation and layer quantisation
  - GPTQ Hessian construction plus damped Cholesky inversion
  - GGUF Q4_K_M block quant/dequant helpers
  - QAT fake-quant nodes
  - INT8 `QuantisedKvCache`
  - Streaming calibration stats over real model linear inputs

- **`aarambh-ai-weights`**
  - Added GGUF save/load support for Q4_K_M, Q5_K_M, and Q8_0 formats
  - Added `.gguf` model loading through `load_any_model()`
  - Implemented HuggingFace safetensors conversion with standard key mapping and strict GQA K/V slicing

- **CLI**
  - Added `aarambh-ai quantise`
  - Added `aarambh-ai convert`
  - Added `aarambh-ai convert --gguf`
  - `aarambh-ai infer --model <path.gguf>` now loads GGUF checkpoints

### Changed

- **`aarambh-ai-model` / `aarambh-ai-nn`**
  - Added capture-aware forward methods for calibration inputs to attention and FFN linear layers

### Verified

- `cargo check -p aarambh-ai-quant`
- `cargo check -p aarambh-ai-weights`
- `cargo check -p aarambh-ai --all-targets`
- `cargo test -p aarambh-ai-quant`
- `cargo test -p aarambh-ai-weights`

## [0.7.0] - 2026-06-28

### Added

- **Thinking engine (Phase 7)**
  - `ThinkingController` now forces `<think>` once for `low`, `medium`, and `high` modes
  - Enforces mode budgets and force-injects `</think>` when the active budget is reached
  - Tracks started/closed state, effective generation budget, thinking-token count, and queued forced tokens
  - Added `GenerationPhase::{Thinking, Answer}` plus `forced` and `phase` metadata on each generation step

- **Inference output separation**
  - `GenerationOutput.text` is now the visible answer text
  - Added `raw_text`, `thinking_text`, `answer_text`, `thinking_token_ids`, `answer_token_ids`, and `thinking_tokens`
  - Preserves all forced tokens in `token_ids` while hiding thinking markers from user-visible answer output

- **CLI**
  - `aarambh-ai infer --thinking low|medium|high` now wraps prompts with user/assistant markers, prints thinking dimmed, and prints the final answer normally
  - Streaming output switches terminal styling between thinking and answer phases
  - Predict-view now shows token phase and forced-token metadata

- **`aarambh-ai-finetune`**
  - Added `ThinkingSftExample` and `format_thinking_sft()` as the Phase 9-compatible thinking SFT data format stub

### Changed

- **Documentation**
  - Marked Phase 7 complete in README and ROADMAP
  - Updated ARCHITECTURE to describe the implemented thinking controller and separated inference output

### Verified

- `cargo fmt --all -- --check`
- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo run --release -p aarambh-ai -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "What is 2 + 2?" --max-tokens 48 --thinking low --greedy`
- `cargo run --release -p aarambh-ai -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "What is 2 + 2?" --max-tokens 40 --thinking low --greedy --stream`
- `cargo run --release -p aarambh-ai -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "What is 2 + 2?" --max-tokens 6 --thinking low --greedy --predict-view`

## [0.6.0] - 2026-06-28

### Added

- **`aarambh-ai-inference` crate** — Inference engine (Phase 6)
  - `InferenceEngine` with checkpoint loading, tokenizer validation, prompt prefill, cached one-token decode, EOS/max-token/context-limit stopping, and callback-based generation
  - `KvCache` wrapper over per-layer `aarambh-ai-nn::KVCache`
  - `Sampler` with greedy decode plus temperature/top-k/top-p sampling and top-candidate reporting for predict-view
  - `GenerationConfig`, `GenerationOutput`, `GenerationStep`, `FinishReason`, and `StreamEvent`
  - `ThinkingMode` and `ThinkingController` stub for Phase 7 budget tracking without token forcing

- **CLI**
  - Added `aarambh-ai infer` with `--config`, `--model`, `--tokenizer`, `--prompt`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--thinking`, `--predict-view`, `--stream`, and `--greedy`
  - Defaults to `latest.json` or `best.json` from the configured checkpoint directory when `--model` is omitted
  - Added terminal predict-view rendering for top next-token candidates

- **Tokenizer**
  - Trained BPE tokenizers now reserve fixed project special-token IDs 0..6
  - Added special-token validation and special-aware encode support for `<|endoftext|>`, `<|pad|>`, `<|bos|>`, `<think>`, `</think>`, `<|user|>`, and `<|assistant|>`
  - Training automatically regenerates an owned stale tokenizer whose reserved IDs are invalid

### Changed

- **Binary crate structure**
  - Split CLI implementation into `cmd/train.rs`, `cmd/infer.rs`, and `ui/predict_view.rs`

- **Documentation**
  - Marked Phase 6 complete in README and ROADMAP
  - Updated ARCHITECTURE with tokenizer special-ID invariants and the implemented inference flow

### Verified

- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo run --release -p aarambh-ai -- train --config configs/tiny_shakespeare_smoke.toml`
- `cargo run --release -p aarambh-ai -- infer --config configs/tiny_shakespeare_smoke.toml --prompt "To be" --max-tokens 8 --greedy --predict-view`

## [0.5.0] - 2026-06-27

### Added

- **`aarambh-ai-train` crate** — Training loop (Phase 5)
  - Masked cross-entropy over `[batch, seq, vocab]` logits with padding masks
  - Project-owned AdamW with `beta1=0.9`, `beta2=0.95`, `eps=1e-8`, decoupled weight decay, and no-decay exclusions for embeddings, biases, and RMSNorm weights
  - Explicit gradient accumulation by parameter name, global norm clipping, cosine schedule with linear warmup, validation, logging, and full train loop
  - SafeTensors checkpointing for model weights and optimizer moments plus JSON train state, `latest.json`, and `best.json`
  - TOML run config loader, `configs/tiny_shakespeare.toml`, and `configs/tiny_shakespeare_smoke.toml`
  - 11 train tests covering loss masking, LR warmup/decay, AdamW defaults, weight decay policy, gradient clipping, checkpoint roundtrip, and synthetic tiny-model loss decrease

- **CLI**
  - Added `aarambh-ai train --config <path>` for Phase 5 training runs

- **Tokenizer**
  - Added `BpeTokenizer::save_pretrained()` to persist vocab and BPE merges in a reloadable tokenizer JSON
  - Supports both legacy string merges and modern array merges from HuggingFace `tokenizers`

### Changed

- **`aarambh-ai-core` crate**
  - Extended `TrainConfig` with `max_steps`, `min_lr_ratio`, and `seed`
  - Added serde defaults for backward-compatible config loading

- **`aarambh-ai-nn` and `aarambh-ai-model` crates**
  - Added `forward_train()` paths that use Candle autograd-compatible RMSNorm and attention instead of Phase 4 inference kernels
  - Changed token embedding initialization to `N(0, 0.02)` so tied LM heads start with sane logits and random-model loss near `ln(vocab)`

- **Training config**
  - Reuses an existing tokenizer JSON in the checkpoint directory instead of retraining BPE on every launch

### Verified

- `cargo check --workspace --all-targets`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo run --release -p aarambh-ai -- train --config configs/tiny_shakespeare_smoke.toml`

## [0.4.0] - 2026-06-27

### Added

- **`aarambh-ai-kernel` crate** — Custom kernels (Phase 4)
  - Runtime dispatch API for RMSNorm and attention
  - Stable CPU SIMD RMSNorm with cached AVX2/FMA, AVX512, AVX2, and scalar fallback paths
  - Rayon parallel scaled dot-product attention for CPU F32 tensors
  - Candle fallback for unsupported devices, dtypes, shapes, and CUDA runtime paths
  - `build.rs` NVCC detection with graceful no-CUDA builds
  - CUDA placeholder kernels and FFI wrapper modules for Flash Attention, fused RMSNorm, fused RoPE, and fused SwiGLU
  - Criterion benchmark target for RMSNorm and attention kernels
  - 6 kernel tests covering dispatch, RMSNorm reference parity, parallel attention parity, masks, and CUDA stubs
  - Local benchmark: RMSNorm SIMD ~1.43x faster than Candle; parallel attention ~2.94x faster than sequential

### Changed

- **`aarambh-ai-nn` crate**
  - `RMSNorm::forward()` now calls kernel dispatch
  - `GroupedQueryAttention::forward()` now calls kernel attention dispatch after Q/K/V preparation

- **Documentation**
  - Marked Phase 4 complete in README and ROADMAP
  - Updated ARCHITECTURE to match stable SIMD intrinsics and CUDA stub behavior

## [0.3.0] - 2026-06-27

### Added

- **`aarambh-ai-model` crate** — Full model forward pass (Phase 3)
  - `src/embedding.rs` — `TokenEmbedding` wrapping Candle embedding lookup with weight access for tied LM head
  - `src/head.rs` — `LmHead` supporting tied embedding weights and untied no-bias output projection
  - `src/model.rs` — `AarambhModel` with config validation, embedding, N transformer blocks, final RMSNorm, LM head, precomputed RoPE, precomputed causal mask, full-sequence `forward()`, cached `forward_with_cache()`, `empty_kv_cache()`, `named_tensors()`, and `get_weight()`
  - Implements `Configurable` and `Forward`
  - 8 active integration tests covering scale config validation, Tiny forward shape, finite logits, cached-vs-full forward equivalence, tied/untied LM head behavior, invalid config rejection, and README scale consistency
  - 1 ignored heavy test for full Tiny/Small/Medium/Large construction

- **`aarambh-ai-weights` crate** — SafeTensors I/O (Phase 3)
  - `save_model()` serializes `AarambhModel::named_tensors()` with `candle_core::safetensors::save`
  - `load_model()` loads SafeTensors through `VarBuilder::from_mmaped_safetensors`
  - `convert_hf()` is present as a Phase 8 `Unsupported` stub
  - 2 integration tests covering SafeTensors weight/logit roundtrip and the Phase 8 conversion stub

### Changed

- **`aarambh-ai-nn` crate**
  - Added read-only weight accessors on `GroupedQueryAttention`, `SwiGluFfn`, and `TransformerBlock` so higher layers can enumerate model tensors without making fields public

- **Documentation**
  - Updated README model scale table to match `ModelConfig`, `ARCHITECTURE.md`, and `ROADMAP.md`
  - Marked Phase 3 complete in README and ROADMAP

## [0.2.0] - 2026-06-25

### Added

- **`aarambh-ai-nn` crate** — Neural network primitives (Phase 2)
  - `src/norm.rs` — `RMSNorm` wrapping `candle_nn::ops::rms_norm` with learnable weight
  - `src/rope.rs` — `RopeCache` precomputing cos/sin tables for up to `max_seq_len`, applying rotary position embeddings to Q/K
  - `src/kvcache.rs` — `KVCache` with `update()` (catches K/V along seq dim), `clear()`, `seq_len()`
  - `src/attention.rs` — `GroupedQueryAttention` with Q/K/V projections, RoPE, K/V head expansion for GQA, causal masking, `softmax_last_dim`, output projection
  - `src/ffn.rs` — `SwiGluFfn` with gate/up/down projections and SiLU-gated activation
  - `src/block.rs` — `TransformerBlock` with pre-norm residual connections
  - 5 integration tests covering RMSNorm shape, RoPE magnitude, SwiGLU shape, GQA output, and full block output
  - `src/lib.rs` — flat re-exports of all modules

## [0.1.0] - 2026-06-24

### Added

- **`aarambh-ai-tokenizer` crate** — BPE tokeniser
  - `src/special.rs` — 7 special token ID constants
  - `src/vocab.rs` — `Vocab` struct with `HashMap`-backed token↔id lookup, JSON I/O
  - `src/bpe.rs` — `BpeTokenizer` with `train()` (delegates to `tokenizers` crate BpeTrainer), `from_pretrained()` (parses HuggingFace `tokenizer.json`), pure-Rust `encode()`/`decode()`, `save()`, `TokenizerLike` impl
  - 5 unit tests covering all paths

- **`aarambh-ai-data` crate** — Data pipeline
  - `src/dataset.rs` — `TextDataset` trait, `PlaintextDataset` (`.txt` files), `JsonlDataset` (`.jsonl` with `{"text": "..."}` format)
  - `src/preprocess.rs` — `chunk_and_tokenize(dataset, tokenizer, max_seq_len)` produces `(input, label)` pairs with labels shifted by 1
  - `src/loader.rs` — `Batch` struct (input_ids, labels, attention_mask tensors), `DataLoader` struct implementing `Iterator<Item=Result<Batch>>` with batching, shuffling, device placement, and epoch `reset()`
  - 8 unit tests

### Changed

- Updated all 14 crate `Cargo.toml` files to `edition = "2024"`

## [0.0.1] - 2026-06-24

### Added

- **Workspace scaffold**
  - Root `Cargo.toml` with all 14 workspace members and pinned dependency versions
  - `resolver = "2"` for modern feature resolution
  - Workspace-level dependencies: `candle-core`, `candle-nn`, `tokenizers`, `serde`, `thiserror`, `tokio`, `clap`, `tracing`, `safetensors`, `rayon`, `cc`, `which`

- **`aarambh-ai-core` crate** (Layer 0 — Foundation types)
  - `config.rs` — `ModelConfig` with `tiny()`/`small()`/`medium()`/`large()` presets, `head_dim()`, `from_json()`; `TrainConfig` with LLaMA-correct defaults (`beta2=0.95`, `batch_size=2`, `grad_accum_steps=16`)
  - `device.rs` — `Device` enum (`Cpu`, `Cuda`, `Metal`) with `to_candle()`, `best_available()`, `is_cpu()`
  - `dtype.rs` — `DType` (`F32`, `F16`, `BF16`) with `to_candle()`, `size_bytes()`; `Precision` with `weight_dtype()`
  - `error.rs` — `AarambhError` (8 variants using `thiserror`), `type Result<T>`
  - `traits.rs` — `Forward`, `Saveable`, `Loadable`, `Configurable`, `TokenizerLike`
  - `lib.rs` — flat re-exports of all public types
  - `tests/core_tests.rs` — 6 unit tests covering configs, device, dtype, and defaults

- **12 stub crates** — each with `Cargo.toml` + `lib.rs` doc-comment placeholder
  - `aarambh-ai-tokenizer`, `aarambh-ai-data`, `aarambh-ai-nn`, `aarambh-ai-kernel`, `aarambh-ai-model`, `aarambh-ai-weights`, `aarambh-ai-quant`, `aarambh-ai-train`, `aarambh-ai-finetune`, `aarambh-ai-inference`, `aarambh-ai-safety`, `aarambh-ai-selflearn`

- **Binary crate** — `aarambh-ai` with minimal `main.rs`

- **GitHub repository files**
  - `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
  - `LICENSE` (Apache 2.0)
  - `.gitignore`, `.github/` (CI workflow, issue/PR templates, dependabot)
