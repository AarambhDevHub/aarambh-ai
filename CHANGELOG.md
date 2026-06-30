# Changelog

## [0.14.0] - 2026-06-30

### Added

- **Phase 14 Flash Attention CUDA kernels**
  - Replaced Phase 4 CUDA placeholders with real `.cu` kernels for Flash Attention forward, Flash Attention backward source, fused RMSNorm, fused RoPE, and fused SwiGLU
  - Added NVCC-to-PTX build plumbing with `cfg(aarambh_cuda_kernels)` and graceful CPU/Candle fallback when NVCC is missing
  - Added Candle custom-op wrappers that load PTX into Candle's CUDA module cache at runtime
  - Added CUDA dispatch paths for supported contiguous F32/F16/BF16 FlashAttention and fused RMSNorm tensors
  - Added inference-only fused RoPE and fused SwiGLU hooks in `aarambh-ai-nn`
  - Added CUDA-gated kernel correctness tests against Candle references

### Changed

- **Kernel dispatch**
  - `KernelPath` now reports `CudaFlashAttention` and `CudaFusedRmsNorm` when CUDA PTX kernels are compiled and tensor shapes are supported
  - Attention dispatch detects project causal masks and routes supported CUDA cases to FlashAttention; arbitrary additive masks keep using Candle
  - Training attention uses a dedicated dispatch entry with Candle-compatible backward fallback behavior

- **Documentation**
  - Marked Phase 14 complete in README and ROADMAP
  - Updated architecture notes to describe PTX loading instead of CUDA stubs

### Verified

- `cargo fmt`
- `cargo check`
- `cargo test -p aarambh-ai-kernel`
- `cargo test -p aarambh-ai-nn`
- CUDA PTX tests are gated and must be run on a CUDA host with NVCC and `--features cuda`

## [0.13.0] - 2026-06-29

### Added

- **Phase 13 GPU scale-up implementation**
  - Added opt-in `cuda` feature forwarding across the workspace without changing default CPU builds
  - Added config-driven `dtype = "f32"|"f16"|"bf16"|"mixed"` parsing for train/infer paths
  - Added dtype-aware SafeTensors/GGUF model loading helpers for BF16 GPU inference and self-learning
  - Added WikiText-103 Small, Medium, Large, and CUDA smoke configs
  - Added Kaggle notebooks for Small/T4, Medium/P100, and Large/A100 training workflows
  - Added WikiText-103 preparation and checkpoint packaging scripts

### Changed

- **Training**
  - Trainer now builds model weights using the configured dtype instead of hardcoded F32
  - Cross-entropy casts logits to F32 for stable lower-precision training
  - AdamW keeps moment buffers and update math in F32 while writing params back to their model dtype
  - Training logs now include `tok/s` throughput for Phase 13 benchmarking

- **Model internals**
  - RoPE caches and causal masks are dtype-aware for BF16 model execution
  - Inference and self-learning model loaders now honor the run config dtype

- **Documentation**
  - Marked Phase 13 complete in README and ROADMAP
  - Updated architecture notes with CUDA feature commands and BF16 config behavior

### Verified

- `cargo fmt --all --check`
- `cargo check --workspace`
- `cargo test --workspace`
- `cargo test -p aarambh-ai-train`
- `cargo clippy --workspace --all-targets -- -D warnings`
- Phase 13 notebook JSON validation
- Phase 13 helper script syntax and dummy runtime checks
- CUDA training is prepared through notebooks/configs and must be executed on Kaggle or another CUDA host with `--features cuda`

## [0.12.0] - 2026-06-29

### Added

- **`aarambh-ai-selflearn` Phase 12 implementation**
  - Added CPU/GPU/disabled self-learning presets with online GRPO, replay, critique, metrics, and persistent state configuration
  - Added replay buffer JSONL persistence, score filtering, high-quality retention, score-squared sampling, topic diversity, and topic inference
  - Added stateless replay-only self-critique with robust JSON parsing, score clamping, malformed-output fallback, bounded rewrite generation, and rewrite support
  - Added LoRA-backed self-learning generation, deterministic-verifier online GRPO updates, CPU deferred pending gradients, GPU inline stepping, optimizer state, adapter state, pending-gradient persistence, and pending-gradient contribution accounting
  - Added replay SFT updates that train the live LoRA adapter from sampled replay entries with response-only loss masking
  - Added learning metrics with per-topic trend summaries

### Changed

- **CLI**
  - Added `--self-learn disabled|cpu|gpu`, `--replay-path`, `--self-learn-state-dir`, `--self-learn-verifier`, and `--self-learn-ground-truth` to `infer`
  - Added `aarambh-ai selflearn flush-gradients`, `replay`, `stats`, and `reset --yes`
  - Composed self-learning with safety so replay/gradient state commits only after safety allows the generated draft

- **Documentation**
  - Marked Phase 12 complete in README and ROADMAP
  - Updated self-learning CLI examples and architecture notes

### Verified

- `cargo check --workspace`
- `cargo test -p aarambh-ai-selflearn`
- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo run -p aarambh-ai -- infer --help`
- `cargo run -p aarambh-ai -- selflearn --help`
- `cargo run -p aarambh-ai -- selflearn replay --help`
- `cargo run -p aarambh-ai -- selflearn stats --replay-path /tmp/aarambh_phase12_empty_replay.jsonl --self-learn-state-dir /tmp/aarambh_phase12_empty_state`

## [0.11.0] - 2026-06-29

### Added

- **`aarambh-ai-safety` Phase 11 implementation**
  - Added prompt-injection and jailbreak detectors with weighted rule scoring, role-switch checks, leetspeak/confusable normalization, and Base64-like payload detection
  - Added PII detection/redaction for email, phone, SSN/national ID, credit cards with Luhn validation, known API-key prefixes, and high-entropy secrets
  - Added output toxicity scoring for hate speech, violence, sexual content, self-harm, and illegal activity
  - Added `SafetyPolicy` presets, `SafetyVerdict`, `SafetyGuard`, `SafeResponse`, and privacy-safe `SafetyEvent` audit logging with SHA-256 prompt hashes

### Changed

- **CLI**
  - `infer` now uses `SafetyGuard` by default
  - Added `--safety strict|permissive|research|none` and `--safety-audit-log`
  - Buffered safety-enabled streaming/predict-view callbacks until output checks pass, preventing unsafe text from being printed before guardrails run

- **Documentation**
  - Marked Phase 11 complete in README and ROADMAP
  - Added safety CLI examples and audit privacy notes
  - Aligned ARCHITECTURE safety policy fields with the implemented API

### Verified

- `cargo check -p aarambh-ai-safety`
- `cargo check -p aarambh-ai`
- `cargo test -p aarambh-ai-safety`

## [0.10.0] - 2026-06-29

### Added

- **`aarambh-ai-finetune` Phase 10 implementation**
  - Added GRPO dataset loading for `prompt`/`question` plus `ground_truth`/`answer` JSONL records
  - Added graph-free group rollout sampling from the live LoRA policy with temperature, top-k, top-p, and thinking-token forcing
  - Added differentiable replay of sampled completions through `LoraAarambhModel::forward_train()` for policy log-probs
  - Added frozen-reference KL loss using full generated-token distributions
  - Added `GrpoTrainer` with adapter-only AdamW, cosine warmup/decay, gradient accumulation, clipping, logging, and adapter checkpoints

- **Deterministic verifiers**
  - Added `MathVerifier` with GSM8K `#### answer` parsing, commas, negatives, decimals, and numeric tolerance
  - Added `FormatVerifier` for `<think>...</think>` structure rewards
  - Added `CompositeVerifier` and `math-format` verifier selection

- **CLI**
  - Added `aarambh-ai finetune grpo`
  - Added GRPO flags for reference checkpoint, verifier, group size, max new tokens, sampling controls, thinking mode, KL coefficient, LoRA rank/alpha/dropout, steps, LR, accumulation, logging, and save cadence

### Changed

- **Documentation**
  - Marked Phase 10 complete in README and ROADMAP
  - Added GRPO command examples, tiny local GRPO math data, and adapter output notes
  - Clarified that GRPO uses deterministic verifiers only; Self-Critique remains Phase 12 replay-buffer logic
  - Documented that GRPO training log-probs are recomputed by differentiable replay instead of cached inference

### Verified

- `cargo check -p aarambh-ai-finetune -p aarambh-ai`
- `cargo check --workspace`
- `cargo test -p aarambh-ai-finetune`
- `cargo test --workspace`

## [0.9.0] - 2026-06-29

### Added

- **`aarambh-ai-finetune` Phase 9 implementation**
  - Added `LoraConfig`, `LoraLinear`, frozen F32 base support, packed INT4 QLoRA base support, adapter dropout, target-module matching, and LoRA merge math
  - Added `LoraAarambhModel`, an adapter-aware decoder forward path that keeps the existing base model/inference code unchanged
  - Added adapter persistence with `adapter_config.json`, `adapter.safetensors`, and adapter train-state output
  - Added SFT JSONL loading for `{"instruction","response"}` and `{"instruction","thinking","response"}`
  - Added chat templates, thinking SFT formatting, shifted labels, prompt loss masking, and padded SFT batches
  - Added `SftTrainer` with adapter-only AdamW, cosine warmup/decay, gradient accumulation, clipping, logging, and adapter checkpoints
  - Added adapter merge into normal `model.safetensors` for the existing inference engine

- **CLI**
  - Added `aarambh-ai finetune sft`
  - Added `aarambh-ai finetune qlora`
  - Added `aarambh-ai finetune merge`
  - Added fine-tune overrides for LoRA rank/alpha/dropout, target modules, batch size, max steps, learning rate, accumulation, warmup, logging, and save cadence

### Changed

- **Documentation**
  - Marked Phase 9 complete in README and ROADMAP
  - Added LoRA/QLoRA/SFT command examples and adapter layout documentation
  - Clarified that DoRA is not part of Phase 9

### Verified

- `cargo fmt`
- `cargo check --workspace`
- `cargo test --workspace`

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
