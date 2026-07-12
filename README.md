# aarambh-ai

> **Sanskrit: *beginning* — A ground-up LLM in Rust**

[![CI](https://github.com/AarambhDevHub/aarambh-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/AarambhDevHub/aarambh-ai/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.89%2B-orange.svg)](https://www.rust-lang.org)

A decoder-only transformer with four model scales, a three-level thinking engine, full training pipeline, quantisation (INT8/INT4/GGUF), LoRA/QLoRA/DoRA fine-tuning, GRPO and DPO alignment, exact speculative decoding, grammar-constrained function calling, custom CUDA + SIMD kernels, safety guardrails, self-learning loop, evaluation harness, a frozen-encoder vision projector path, and an OpenAI-compatible inference server — all in one clean 17-crate Rust workspace.

v2.0.0 is a production GitHub source release. The workspace crates are internal
implementation units and remain non-publishable; no pretrained checkpoints or
compiled binaries are attached to the release.

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
| Thinking engine: Low / Medium / High reasoning budgets | Phase 7 ✅ |
| Full training pipeline with AdamW, cosine schedule, checkpointing | Phase 5 ✅ |
| Quantisation: INT8, GPTQ INT4, AWQ INT4, GGUF, QAT | Phase 8 ✅ |
| LoRA, QLoRA, SFT fine-tuning | Phase 9 ✅ |
| DoRA, QDoRA weight-decomposed fine-tuning | Phase 18 ✅ |
| GRPO reinforcement learning | Phase 10 ✅ |
| Safety guardrails: input/output, PII, prompt injection | Phase 11 ✅ |
| Self-learning loop: online GRPO, replay buffer, critique | Phase 12 ✅ |
| GPU scale-up: CUDA feature, BF16 train/infer, Kaggle configs/notebooks | Phase 13 ✅ |
| Custom CUDA kernels: Flash Attention v2, fused RMSNorm, RoPE, SwiGLU | Phase 14 ✅ |
| Long context: YaRN/NTK/linear RoPE scaling, 16K configs, staged continuation | Phase 16 ✅ |
| CPU SIMD kernels: AVX2/FMA RMSNorm, AVX512 override, parallel attention via rayon | Phase 4 ✅ |
| CUDA kernel build prep | Phase 4 ✅ |
| CLI binary with predict-view, streaming, thinking modes | Phase 6 ✅ |
| Production v1.0 source release: strict docs, CI, release workflow, release notes | Phase 15 ✅ |
| Evaluation harness: PPL, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite scorecards | Phase 17 ✅ |
| Vision encoder + projector: CLIP ViT, image preprocessing, projector pretrain, `infer --image` | Phase 19 ✅ |
| Vision-language instruction tuning: VQA JSONL/LLaVA data, VLM DoRA/QDoRA, VQA eval | Phase 20 ✅ |
| Vision-aware self-learning: image replay cache, grounded VQA verifier, CUDA gate | Phase 21 ✅ |
| Mixture of Experts: top-k router, dense masked dispatch, load-balanced training | Phase 22 ✅ |
| Multi-GPU training: single-node NCCL data parallelism, sharded loaders, rank-0 checkpoints | Phase 23 ✅ |
| DPO/QDPO preference tuning: cached references, pairwise loss, preference win-rate eval | Phase 24 ✅ |
| Exact speculative decoding: Tiny draft, block verification, rejection sampling, telemetry | Phase 25 ✅ |
| Tool use: schema-constrained JSON calls, Tool SFT/QLoRA, selection evaluation | Phase 26 ✅ |
| OpenAI-compatible Axum HTTP/SSE server with continuous batching | Phase 27 ✅ |
| Production v2.0 source release: locked dependencies, strict docs, CI and release audit | Phase 28 ✅ |

---

## Quick Start

### Prerequisites

- Rust 1.89+ ([install via rustup](https://rustup.rs/))
- No GPU required for development (Tiny trains on any i3 laptop)

### Build & Test

```sh
git clone https://github.com/AarambhDevHub/aarambh-ai.git
cd aarambh-ai

# Check the entire workspace compiles
cargo check --workspace --all-targets --locked

# Run all tests
cargo test --workspace --locked

# Build a release binary
cargo build --release -p aarambh-ai --locked

# Run the CLI
cargo run --release --locked -p aarambh-ai -- --help

# Optional local install from this source checkout
cargo install --path aarambh-ai --locked
aarambh-ai --version
```

---

## Production v2.0 Source Release

aarambh-ai v2.0.0 is released as source through GitHub tags and release notes:

- Build/install from the repository source tree.
- All 17 packages use `publish = false`; Aarambh AI is distributed as an application, not as crates.io libraries.
- No pretrained checkpoints, model weights, adapters, tokenizer artifacts, GGUF files, or compiled binaries are released.
- `Cargo.lock` is committed and all release commands use `--locked`.
- Example Tiny configs and data paths are for local smoke tests and user-created checkpoints.
- CUDA is optional; default CPU builds work without NVCC.

```sh
git clone https://github.com/AarambhDevHub/aarambh-ai.git
cd aarambh-ai
git checkout v2.0.0
cargo build --release -p aarambh-ai --locked
target/release/aarambh-ai --version
```

---

## Train Tiny

Phase 5 adds a working training loop for Tiny-scale pretraining:

```sh
# Put Tiny Shakespeare at data/tiny_shakespeare.txt first.
cargo run --release -p aarambh-ai -- train --config configs/tiny_shakespeare.toml

# Fast CPU smoke run for checking the training path.
cargo run --release -p aarambh-ai -- train --config configs/tiny_shakespeare_smoke.toml
```

The trainer builds or loads a BPE tokenizer, creates train/validation loaders,
uses an autograd-safe model forward path, applies masked cross-entropy, AdamW
with `beta2=0.95`, cosine warmup/decay, gradient clipping, gradient
accumulation, and checkpoint save/resume. If a configured tokenizer already
exists and has the required reserved special-token IDs, the trainer reuses it
instead of retraining BPE on every launch; stale Phase 5 tokenizers are
regenerated automatically when the config owns the tokenizer path.

Checkpoint layout:

```text
checkpoints/tiny_shakespeare/
├── latest.json
├── best.json
├── tokenizer.json
├── step_001000/
│   ├── model.safetensors
│   ├── optimizer.safetensors
│   └── train_state.json
└── best/
    ├── model.safetensors
    ├── optimizer.safetensors
    └── train_state.json
```

Typical log lines:

```text
step=1 loss=9.0304 ppl=8352.87 lr=0.000250 grad_norm=0.7182
step=10 loss=9.0241 ppl=8300.43 lr=0.000800 grad_norm=0.7221 tok/s=182.44
eval step=500 val_loss=3.2110 val_ppl=24.80
```

---

## Train On Kaggle GPU

Phase 13 adds an opt-in CUDA/BF16 path for WikiText-103 scale-up. Phase 14 adds
runtime PTX kernels for Flash Attention, fused RMSNorm, fused RoPE, and fused
SwiGLU when `nvcc` is available at build time. CUDA is not enabled by default,
so normal local CPU builds still work without NVCC.

```sh
# Prepare WikiText-103 raw text.
scripts/phase13_prepare_wikitext103.sh data

# Fast CUDA/BF16 smoke test on Kaggle.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_cuda_smoke.toml

# Tiny on Kaggle GPU.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_tiny.toml

# Small on T4.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_small.toml

# Medium on P100.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_medium.toml

# Large on A100.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_large.toml
```

GPU configs use `device = "cuda:0"` and `dtype = "bf16"`. Training logs include
`tok/s` so Kaggle runs produce the Phase 13 throughput benchmark automatically.
On CUDA builds with NVCC, Phase 14 dispatch chooses CUDA fused RMSNorm and
FlashAttention for supported contiguous F32/F16/BF16 tensors; unsupported masks,
layouts, and dtypes fall back to Candle.
Use `notebooks/phase13_small_train.ipynb`, `phase13_medium_train.ipynb`, and
`phase13_large_train.ipynb` for end-to-end Kaggle runs, inference smoke checks,
and packaged checkpoint downloads.

```sh
# Package a trained checkpoint directory for download.
scripts/phase13_pack_checkpoint.sh checkpoints/wikitext103_small phase13_small_checkpoint.zip
```

---

## Multi-GPU Training

Phase 23 adds single-node data-parallel training for Kaggle 2×T4 sessions. It
uses one worker process per GPU, NCCL all-reduce for averaged gradients,
deterministic rank-local data shards, and rank-0-only logging/checkpointing.
When only one GPU is visible, rank 0 falls back to the normal single-process
path and nonzero workers exit cleanly.

```sh
cargo build --release -p aarambh-ai --features cuda

export AARAMBH_WORLD_SIZE=2
export AARAMBH_DIST_RUN_ID=wikitext-2gpu-$(date +%s)

AARAMBH_RANK=0 AARAMBH_LOCAL_RANK=0 ./target/release/aarambh-ai train \
  --config configs/wikitext103_small_2gpu.toml &
AARAMBH_RANK=1 AARAMBH_LOCAL_RANK=1 ./target/release/aarambh-ai train \
  --config configs/wikitext103_small_2gpu.toml &
wait
```

The `[distributed]` TOML section can provide defaults, but
`AARAMBH_WORLD_SIZE`, `AARAMBH_RANK`, `AARAMBH_LOCAL_RANK`,
`AARAMBH_DIST_RUN_ID`, and `AARAMBH_DIST_RENDEZVOUS` override it at runtime.
Phase 23 is single-node only; multi-node rendezvous is intentionally out of
scope.

---

## Long Context 16K

Phase 16 adds RoPE scaling for continued pretraining at longer context lengths.
The default v1 configs keep `rope_scaling = None`, so unscaled outputs remain
compatible. Long-context configs set `model.rope_scaling` and use staged loader
growth so training warms up at 4K, then 8K, then 16K.

```sh
# Prepare WikiText-103 and concatenate it into longer documents.
scripts/phase16_prepare_longdoc.sh data

# Fast long-context CUDA smoke test.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_long_smoke.toml

# Medium 16K continued pretraining.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/medium_16k.toml

# Large 16K continued pretraining.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/large_16k.toml
```

The long-context configs do not ship pretrained checkpoints. They are recipes
for user-run continuation from locally trained or converted model weights.

---

## Evaluation Harness

Phase 17 adds `aarambh-ai eval` for comparable before/after model quality
tracking. It reports JSON and Markdown scorecards for perplexity,
MMLU-lite, HellaSwag, GSM8K-subset, HumanEval-lite, pairwise preference
win rate, and the Phase 19 image-caption smoke task.

```sh
# Prepare public eval subsets. Requires Python's datasets package.
scripts/phase17_prepare_eval_sets.sh data/eval 128

# Run PPL and multiple-choice tasks.
cargo run --release -p aarambh-ai -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks ppl,mmlu,hellaswag \
  --data-dir data/eval \
  --out scorecard.json \
  --markdown scorecard.md

# Compare two scorecards.
cargo run --release -p aarambh-ai -- eval \
  --compare scorecard_before.json scorecard_after.json \
  --markdown compare.md
```

HumanEval-lite executes generated Python tests and is disabled unless
`--allow-code-exec` is passed explicitly.

---

## Vision Projector

Phase 19 adds `aarambh-ai-vision`: local image decode/resize/normalize,
CLIP-B/32 SafeTensors loading, a frozen ViT encoder, and a trainable projector
that turns image patches into LLM-space prefix tokens.

```sh
# Download public CLIP-B/32 SafeTensors and write the matching config.
scripts/phase19_download_clip_weights.sh data/vision

# Prepare a COCO caption JSONL subset and images.
scripts/phase19_prepare_coco_captions.sh data

# Train only the projector; the language model and vision encoder stay frozen.
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/vision_projector_pretrain.toml

# Generate from an image.
cargo run --release -p aarambh-ai --features cuda -- infer \
  --config configs/vision_projector_pretrain.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --image data/sample.jpg \
  --prompt "What is this?" \
  --max-tokens 64

# Run the image-caption eval smoke.
cargo run --release -p aarambh-ai -- eval \
  --config configs/vision_projector_pretrain.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks image-caption \
  --data-dir data/eval
```

Vision tokenizers must reserve `<image>` and `<image_end>` at IDs 7 and 8.
Legacy text-only tokenizers still load for text inference, but `--image`
requires the v2 multimodal special tokens.

## Vision-Language Instruction Tuning

Phase 20 adds VQA-style instruction tuning on top of the Phase 19 projector.
The CLIP encoder stays frozen, the LLM trains through DoRA/QDoRA adapters,
and the projector can either continue training or stay frozen.

```sh
# Local synthetic smoke fixture: tiny CLIP, four images, VQA JSONL, tokenizer,
# and an initial projector checkpoint.
python3 scripts/phase20_make_vqa_smoke_fixture.py

# CPU smoke run.
cargo run --release -p aarambh-ai -- finetune vlm-dora \
  --config configs/vision_vqa_smoke.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/vision_projector_smoke/tokenizer.json \
  --data data/vision_smoke/vqa_smoke_4.jsonl \
  --output adapters/vision_vqa_smoke \
  --projector data/vision_smoke/projector_init.safetensors \
  --lora-rank 4 \
  --max-steps 2 \
  --log-every-n-steps 1

# Merge the DoRA LLM adapter, then point the vision config at the tuned projector.
cargo run --release -p aarambh-ai -- finetune merge \
  --config configs/vision_vqa_smoke.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --adapter adapters/vision_vqa_smoke \
  --method dora \
  --output checkpoints/vision_vqa_smoke_merged

cargo run --release -p aarambh-ai -- eval \
  --config configs/vision_vqa_smoke.toml \
  --model checkpoints/vision_vqa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/vision_projector_smoke/tokenizer.json \
  --tasks vqa \
  --data-dir data/eval \
  --max-examples 2
```

For full Kaggle runs, prepare LLaVA-Instruct metadata and place the matching
COCO images under the configured image root:

```sh
MAX_EXAMPLES=10000 scripts/phase20_prepare_llava_instruct.sh data

cargo run --release --features cuda -p aarambh-ai -- finetune vlm-dora \
  --config configs/vision_vqa_instruct.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --data data/llava/llava_instruct_150k.jsonl \
  --output adapters/vision_vqa_dora \
  --projector checkpoints/vision_projector/step_010000/model.safetensors
```

---

## Infer Tiny

Phase 6 adds a checkpoint-backed inference engine and `infer` CLI command:

```sh
# Use latest.json or best.json from the config checkpoint directory.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare_smoke.toml \
  --prompt "To be, or not to be" \
  --max-tokens 32 \
  --greedy \
  --predict-view

# Stream sampled text from an explicit model/tokenizer pair.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "The king" \
  --max-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50 \
  --stream

# Force a budget-controlled thinking block before the visible answer.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --prompt "What is 15 x 27?" \
  --max-tokens 96 \
  --thinking low \
  --greedy
```

The inference path validates tokenizer special IDs before loading the model,
prefills the prompt, decodes one token at a time with the KV cache, supports
greedy or top-k/top-p sampling, stops on `<|endoftext|>` or context limit, and
can render a next-token predict-view for debugging. `--thinking low|medium|high`
wraps the prompt with user/assistant markers, forces `<think>` as the first
generated token, enforces the mode budget, force-closes with `</think>` when
needed, and prints the final answer separately from the dimmed thinking block.
Reasoning quality still depends on Phase 9/10 thinking SFT and GRPO training.

Phase 25 adds exact speculative decoding for text generation. The draft and
target models may use different architectures but must use identical tokenizer
vocabularies, merge order, and special-token IDs. The target config controls the
device and dtype; `--draft-config` supplies the draft model architecture.

```sh
cargo run --release --features cuda -p aarambh-ai -- infer \
  --config configs/wikitext103_large.toml \
  --model checkpoints/large/model.safetensors \
  --tokenizer checkpoints/shared/tokenizer.json \
  --prompt "The future of efficient inference" \
  --max-tokens 128 \
  --temperature 0.7 --top-p 0.9 --top-k 50 \
  --speculative \
  --draft-config configs/wikitext103_tiny.toml \
  --draft-model checkpoints/tiny/model.safetensors \
  --draft-tokens 4 \
  --stream --stats --safety none
```

Only committed tokens reach streaming and predict-view callbacks. Safety modes
remain supported and retain their buffered-output behavior. Phase 25 is text
only: `--image` and `--self-learn` combinations return explicit errors. Both
SafeTensors and Aarambh GGUF files use the existing universal loader, although
the GGUF path currently dequantises tensors at load time.

Benchmark a Tiny draft against a Medium/Large target on Kaggle after building
with CUDA. Greedy mode verifies byte-identical output while the script reports
the average speed and acceptance rate; the 1.8x roadmap target is not a CI gate.

```sh
RUNS=3 MAX_TOKENS=128 DRAFT_TOKENS=4 \
  scripts/phase25_benchmark_speculative.sh \
  configs/wikitext103_large.toml checkpoints/large/model.safetensors \
  checkpoints/shared/tokenizer.json \
  configs/wikitext103_tiny.toml checkpoints/tiny/model.safetensors
```

### Tool Use / Function Calling

Phase 26 lets the model choose between a direct answer and one structured tool
call. Tool calls are constrained during sampling and validated again before
being returned. Aarambh AI emits the call but does not execute commands, HTTP
requests, or other host actions.

```sh
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tools data/tools_smoke.json \
  --tool-choice auto \
  --prompt "What is 17 multiplied by 23?" \
  --max-tokens 96 \
  --greedy \
  --safety none
```

A completed call is printed as compact JSON and ends with
`finish_reason=ToolCall`:

```json
{"name":"calculate","arguments":{"expression":"17 * 23"}}
```

`--tool-choice required` prevents direct answers; `--tool-choice <name>` pins
one function; `none` forces the direct-answer branch. The tool path composes
with thinking, safety, streaming, predict-view, and speculative decoding.
Streaming buffers tool JSON until it is complete. `--image` and
`--self-learn` combinations are intentionally rejected in Phase 26.

The supported schema subset covers nested objects and arrays, required and
optional properties, strings, numbers, integers, booleans, null, scalar
`enum`/`const`, nullable types, length/item limits, and numeric bounds.
Recursive references, schema composition, regex patterns, conditionals, and
parallel calls are rejected while loading the tools file.

Train a LoRA tool adapter with the tracked four-example smoke set:

```sh
cargo run --release -p aarambh-ai -- finetune tool-sft \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/tool_sft_tiny.jsonl \
  --output adapters/tool_sft_smoke \
  --lora-rank 4 --batch-size 1 --max-steps 2 \
  --log-every-n-steps 1 --save-every-n-steps 0
```

Use `finetune tool-qlora` for a quantized base. Larger single-call SFT data can
be normalized from the gated xLAM dataset after accepting its terms:

```sh
MAX_EXAMPLES=10000 scripts/phase26_prepare_xlam.sh
```

Evaluate exact actions, schema validity, tool-name selection, and no-tool
accuracy with:

```sh
cargo run --release -p aarambh-ai -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks tool-calling --data-dir data/eval \
  --max-examples 2 --max-new-tokens 96
```

`scripts/phase26_prepare_bfcl.sh` prepares a pinned BFCL v1.2 single-call
subset for larger held-out runs. Downloaded datasets, adapters, and checkpoints
remain untracked.

### Inference Server

Phase 27 serves one local checkpoint through OpenAI-compatible chat,
completion, model-list, and SSE endpoints. Active requests share batched model
decode passes while retaining independent KV caches, samplers, thinking state,
stop matching, tool grammar, and safety state.

```sh
cargo run --release -p aarambh-ai -- serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --model-id aarambh-tiny \
  --port 8080
```

The default bind is local-only at `127.0.0.1`. A non-loopback bind requires
`AARAMBH_AI_API_KEY`; request bodies then use `Authorization: Bearer ...`.
Safety is strict by default and streaming output passes through a rolling
cross-token scanner before release. See
[the inference server guide](docs/inference-server.md) for curl, OpenAI SDK,
tool-calling, authentication, metrics, and smoke-test examples.

Phase 11 enables the safety layer by default for `infer`. Use
`--safety strict|permissive|research|none` to choose policy behavior and
`--safety-audit-log` to choose the JSONL audit path. Audit records store a
SHA-256 prompt hash and rule IDs only; prompt text, output text, and thinking
text are never written to the log.

```sh
# Default strict safety: injection/jailbreak checks, PII redaction, output checks.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "To be, or not to be" \
  --max-tokens 64 \
  --greedy \
  --safety strict \
  --safety-audit-log safety_audit.jsonl

# Raw legacy inference for benchmarks/debugging.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --prompt "The king" \
  --max-tokens 64 \
  --safety none
```

Phase 12 adds opt-in self-learning for inference. CPU mode keeps updates
deferred in a persistent LoRA state directory; GPU mode can step inline. Safety
still wraps the user-visible output, and self-learning commits replay/gradient
state only after the safety layer allows the draft.

```sh
# CPU-safe self-learning with critique + replay.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "Explain recursion simply." \
  --max-tokens 64 \
  --self-learn cpu \
  --replay-path data/replay.jsonl \
  --self-learn-state-dir adapters/selflearn

# Deterministic-verifier online GRPO when ground truth is available.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "What is 2 + 2?" \
  --max-tokens 32 \
  --self-learn cpu \
  --self-learn-verifier math \
  --self-learn-ground-truth "#### 4"

# Manage persistent self-learning state.
cargo run --release -p aarambh-ai -- selflearn stats --replay-path data/replay.jsonl
cargo run --release -p aarambh-ai -- selflearn flush-gradients \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json
cargo run --release -p aarambh-ai -- selflearn replay \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --replay-path data/replay.jsonl \
  --self-learn-state-dir adapters/selflearn
```

Phase 21 extends self-learning to image-grounded turns on Kaggle/CUDA only.
It caches frozen projected image tokens in the self-learning state directory,
stores `image_ref` in replay JSONL, and uses grounded verifiers for checkable
VQA questions. Text-only self-learning remains CPU-safe and unchanged.

```sh
# Local gate smoke: this should fail clearly on CPU, proving vision mode will not silently OOM.
cargo run --release -p aarambh-ai -- selflearn start \
  --mode vision \
  --config configs/selflearn_vision_smoke.toml \
  --model checkpoints/vision_vqa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/vision_projector_smoke/tokenizer.json \
  --prompt "What color is the square?" \
  --image data/vision_smoke/images/red_square.png \
  --self-learn-vision-verifier color \
  --self-learn-ground-truth red \
  --safety none

# Kaggle/CUDA vision self-learning turn.
cargo run --release --features cuda -p aarambh-ai -- infer \
  --config configs/selflearn_vision.toml \
  --model checkpoints/vision_vqa_merged/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --image data/llava/images/example.jpg \
  --prompt "What color is the car?" \
  --max-tokens 32 \
  --self-learn gpu \
  --self-learn-vision-verifier color \
  --self-learn-ground-truth red \
  --replay-path data/replay_buffer_v2.jsonl \
  --self-learn-state-dir adapters/selflearn_vision

cargo run --release -p aarambh-ai -- selflearn stats \
  --mode vision \
  --replay-path data/replay_buffer_v2.jsonl \
  --self-learn-state-dir adapters/selflearn_vision
```

---

## Mixture Of Experts

Phase 22 adds optional MoE feed-forward layers. Dense configs remain the
default; MoE activates only when `[model.moe]` is present. The first
implementation uses top-k routing, dense masked dispatch, and a
load-balancing auxiliary loss during base-model training.

```sh
# CPU smoke run.
cargo run --release -p aarambh-ai -- train --config configs/moe_smoke.toml

# Kaggle/P100/A100 Small-MoE run.
cargo run --release --features cuda -p aarambh-ai -- train --config configs/small_moe.toml
```

MoE checkpoint tensor names use router/expert paths such as
`blocks.1.ffn.router.weight` and `blocks.1.ffn.experts.0.w_gate.weight`.
LoRA/DoRA fine-tuning for MoE weights is intentionally not enabled in Phase
22; use dense configs for adapter workflows or train the MoE base model
directly.

---

## Quantise And Convert

Phase 8 adds CPU quantisation, GGUF save/load, HuggingFace checkpoint
conversion, QAT primitives, and INT8 KV-cache storage:

```sh
# Export a SafeTensors checkpoint to INT8 GGUF.
cargo run --release -p aarambh-ai -- quantise \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --bits 8 \
  --method int8 \
  --output checkpoints/tiny-q8.gguf

# Calibrate and export an INT4 GGUF checkpoint.
cargo run --release -p aarambh-ai -- quantise \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/best/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --method gptq \
  --bits 4 \
  --calibration-data data/tiny_shakespeare.txt \
  --samples 128 \
  --output checkpoints/tiny-q4.gguf

# Infer directly from GGUF.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny-q4.gguf \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "To be, or not to be" \
  --max-tokens 64 \
  --greedy

# Convert a HuggingFace safetensors directory into Aarambh SafeTensors.
cargo run --release -p aarambh-ai -- convert \
  --config configs/tiny_shakespeare.toml \
  --input /path/to/hf_model \
  --output checkpoints/hf-aarambh.safetensors \
  --arch llama3
```

The GGUF loader rebuilds an `AarambhModel` from dequantised tensors for full
compatibility with the existing inference engine. The on-disk tensors stay
quantised, so Q4 artifacts are much smaller than SafeTensors checkpoints.

---

## Fine-Tune With LoRA, QLoRA, DoRA, QDoRA, GRPO, Or DPO

Phase 9 adds adapter-only SFT for instruction data. Phase 10 adds GRPO
reinforcement learning with deterministic verifiers. Phase 18 adds DoRA/QDoRA
weight-decomposed adapters. Phase 24 adds DPO/QDPO preference tuning for
open-ended response quality. Training updates only adapter tensors, saves a tiny
adapter directory, and can merge adapters back into a normal
`model.safetensors` for existing inference commands.

Input data is JSONL:

```jsonl
{"instruction":"What is 2 + 2?","response":"4"}
{"instruction":"Solve 3 x 7.","thinking":"3 x 7 is repeated addition.","response":"21"}
```

```sh
# LoRA SFT on a SafeTensors base.
cargo run --release -p aarambh-ai -- finetune sft \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --lora-rank 16 \
  --output adapters/tiny_sft

# QLoRA SFT from a GGUF or SafeTensors base.
cargo run --release -p aarambh-ai -- finetune qlora \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/tiny-q4.gguf \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --lora-rank 16 \
  --output adapters/tiny_qlora

# DoRA SFT on a SafeTensors base.
cargo run --release -p aarambh-ai -- finetune dora \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --lora-rank 16 \
  --output adapters/tiny_dora

# QDoRA SFT from a GGUF or SafeTensors base.
cargo run --release -p aarambh-ai -- finetune qdora \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/tiny-q4.gguf \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --lora-rank 16 \
  --output adapters/tiny_qdora

# Merge an adapter into a normal SafeTensors checkpoint.
cargo run --release -p aarambh-ai -- finetune merge \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --adapter adapters/tiny_sft \
  --method auto \
  --output checkpoints/tiny_sft_merged

# Run the merged model with the existing inference engine.
cargo run --release -p aarambh-ai -- infer \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_sft_merged/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "What is 2 + 2?" \
  --thinking low \
  --greedy
```

GRPO data is JSONL with either `prompt` or `question`, plus either
`ground_truth` or `answer`. GSM8K-style `#### final_answer` records are
accepted.

```jsonl
{"question":"What is 2 + 2?","answer":"#### 4"}
{"prompt":"What is 10 - 7?","ground_truth":"3"}
```

```sh
# Fast CPU smoke run.
cargo run --release -p aarambh-ai -- finetune grpo \
  --config configs/tiny_shakespeare_smoke.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --reference checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/grpo_tiny_math.jsonl \
  --verifier math-format \
  --group-size 2 \
  --max-new-tokens 16 \
  --steps 2 \
  --lora-rank 4 \
  --output adapters/tiny_grpo_smoke

# Kaggle-style GRPO run.
cargo run --release -p aarambh-ai -- finetune grpo \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_sft_merged/model.safetensors \
  --reference checkpoints/tiny_sft_merged/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/gsm8k_train.jsonl \
  --verifier math-format \
  --group-size 8 \
  --max-new-tokens 128 \
  --steps 2000 \
  --lr 0.00001 \
  --kl-coeff 0.01 \
  --output adapters/tiny_grpo
```

GRPO uses deterministic `MathVerifier`, `FormatVerifier`, or `math-format`
composite rewards. It does not use Self-Critique; critique is reserved for the
Phase 12 replay buffer.

DPO data contains one shared prompt and a preferred/dispreferred response pair:

```jsonl
{"prompt":"Explain recursion simply.","chosen":"Recursion solves a problem by calling the same function on a smaller input.","rejected":"Recursion is computer magic."}
```

```sh
# Two-step local DoRA-backed DPO smoke run. --reference defaults to --base.
cargo run --release -p aarambh-ai -- finetune dpo \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/dpo_tiny_preferences.jsonl \
  --output adapters/tiny_dpo_smoke \
  --lora-rank 4 \
  --batch-size 2 \
  --max-steps 2 \
  --grad-accum-steps 1 \
  --lr 0.00001 \
  --log-every-n-steps 1 \
  --save-every-n-steps 0

# QDPO uses the same DPO objective with a quantized QDoRA policy base.
cargo run --release -p aarambh-ai -- finetune qdpo \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/tiny-q4.gguf \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/dpo_tiny_preferences.jsonl \
  --output adapters/tiny_qdpo \
  --reference-free \
  --lora-rank 16
```

Standard DPO precomputes frozen-reference sequence log-probabilities once and
then releases the reference model before optimizer steps. `--reference-free`
omits the reference log-ratio. Chosen and rejected responses share one batched
policy forward, and only response tokens contribute to the objective. Use GRPO
for checkable math/code/format rewards and DPO for human or model-ranked
open-ended preferences.

```sh
# Prepare public preference datasets (requires Python's datasets package).
scripts/phase24_prepare_hh_rlhf.sh data/dpo/hh_rlhf
scripts/phase24_prepare_ultrafeedback.sh data/dpo/ultrafeedback

# Evaluate pairwise preference win rate after merging the DPO adapter.
cargo run --release -p aarambh-ai -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_dpo_merged/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks preference \
  --data-dir data/eval
```

Adapter layout:

```text
adapters/tiny_sft/
├── adapter_config.json
├── adapter.safetensors
├── train_state.json
└── checkpoints/
    └── step_000100/
        ├── adapter_config.json
        ├── adapter.safetensors
        └── train_state.json
```

GRPO adapter directories additionally write `grpo_config.json`. DPO/QDPO
directories write `dpo_config.json` with beta, sequence limits, reference mode,
quantized-policy mode, and train settings.

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
├── aarambh-ai-weights/       ← SafeTensors I/O, GGUF save/load, HuggingFace conversion
├── aarambh-ai-quant/         ← INT8, GPTQ, AWQ, GGUF, KV cache quant
├── aarambh-ai-train/         ← Training loop, AdamW, cosine schedule, checkpointing
├── aarambh-ai-finetune/      ← LoRA, QLoRA, DoRA, QDoRA, SFT, GRPO, verifiers
├── aarambh-ai-inference/     ← Inference engine, KV cache, sampler, streaming
├── aarambh-ai-safety/        ← Input/output guardrails, PII, audit
├── aarambh-ai-selflearn/     ← Self-learning loop, replay buffer, critique
├── aarambh-ai-eval/          ← Evaluation harness, scorecards, benchmark tasks
├── aarambh-ai-vision/        ← Frozen CLIP encoder, projector, image fusion
├── aarambh-ai-serve/         ← Axum HTTP/SSE server, continuous batching
└── aarambh-ai/               ← CLI binary (train, infer, serve, quantise, convert, eval)
```

### Dependency Layers

```
Layer 0  aarambh-ai-core
Layer 1  aarambh-ai-tokenizer   aarambh-ai-data
Layer 2  aarambh-ai-nn          aarambh-ai-kernel
Layer 3  aarambh-ai-model       aarambh-ai-weights    aarambh-ai-quant     aarambh-ai-vision
Layer 4  aarambh-ai-train       aarambh-ai-finetune
Layer 5  aarambh-ai-inference   aarambh-ai-safety     aarambh-ai-selflearn  aarambh-ai-eval
Layer 6  aarambh-ai-serve      aarambh-ai (binary)
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

Long-context variants keep the same parameter counts and set scaled context
through config: Medium 16K uses YaRN factor `8.0` from 2K, and Large 16K uses
YaRN factor `4.0` from 4K.

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
│   ├── aarambh-ai-finetune/     ← LoRA, QLoRA, SFT adapters
│   ├── aarambh-ai-inference/    ← Inference engine
│   ├── aarambh-ai-safety/       ← Safety guardrails
│   ├── aarambh-ai-selflearn/    ← Self-learning loop
│   ├── aarambh-ai-eval/         ← Evaluation harness
│   └── aarambh-ai-vision/       ← Vision encoder + projector
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
| 4 | Custom kernels (CPU SIMD + CUDA build prep) | i3 + GPU | ✅ |
| 5 | Training loop — Tiny trains! | i3 | ✅ |
| 6 | Inference engine + CLI | i3 | ✅ |
| 7 | Thinking engine | i3 | ✅ |
| 8 | Quantisation stack | i3 | ✅ |
| 9 | Fine-tuning (LoRA, QLoRA, SFT) | i3 + GPU | ✅ |
| 10 | GRPO reinforcement learning | GPU | ✅ |
| 11 | Safety layer | i3 | ✅ |
| 12 | Self-learning loop | i3 + GPU | ✅ |
| 13 | GPU scale-up (Small → Large) | GPU | ✅ |
| 14 | Flash Attention CUDA kernels | GPU | ✅ |
| 15 | Production release v1.0 | all | ✅ |
| 16 | Long context (RoPE scaling) | i3 + GPU | ✅ |
| 17 | Evaluation harness | i3 + GPU | ✅ |
| 18 | DoRA/QDoRA fine-tuning | i3 + GPU | ✅ |
| 19 | Vision encoder + projector | GPU | ✅ |
| 20 | Vision-language instruction tuning | GPU | ✅ |
| 21 | Vision-aware self-learning | GPU | ✅ |
| 22 | Mixture of Experts | GPU | ✅ |
| 23 | Multi-GPU training | 2×T4 | ✅ |
| 24 | DPO/QDPO preference tuning | GPU | ✅ |
| 25 | Speculative decoding | GPU | ✅ |
| 26 | Tool use / function calling | i3 + GPU | ✅ |
| 27 | OpenAI-compatible inference server | i3 | ✅ |
| 28 | Production release v2.0.0 | all | ✅ |

See [ROADMAP.md](ROADMAP.md) and [ROADMAP_V2.md](ROADMAP_V2.md) for the full phased delivery plans with tests and milestones.

---

## Development Checks

```sh
cargo check --workspace --all-targets --locked
cargo test --workspace --no-fail-fast --locked
cargo clippy --workspace --all-targets --locked -- -D warnings -D clippy::undocumented_unsafe_blocks
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc --workspace --no-deps --locked
scripts/phase28_release_audit.sh
```

### Kernel Benchmarks

```sh
cargo bench -p aarambh-ai-kernel
```

Phase 4 uses stable CPU intrinsics with cached AVX2/FMA, AVX512, and scalar
dispatch. The default prefers AVX2/FMA on this CPU; set `AARAMBH_SIMD_FORCE=avx512`
to force AVX512 when it wins on another machine. Phase 14 CUDA PTX is generated
only when `nvcc` is installed; otherwise the build emits a warning and keeps the
CPU/Candle fallback path.

---

## Documentation

| Document | What it covers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Complete design document, layer-by-layer explanation, data flow, thinking engine, quantisation, fine-tuning, safety, self-learning |
| [ARCHITECTURE_V2.md](ARCHITECTURE_V2.md) | v2 architecture additions: long context, evaluation, vision, MoE, distributed training, tools, and serving |
| [ROADMAP.md](ROADMAP.md) | Completed v1 phased delivery plan |
| [ROADMAP_V2.md](ROADMAP_V2.md) | Completed v2 phased delivery plan through Phase 28 |
| [SELF_LEARNING.md](SELF_LEARNING.md) | Deep dive into the self-learning loop: online GRPO, replay buffer, self-critique, CPU vs GPU modes |
| [SELF_LEARNING_V2.md](SELF_LEARNING_V2.md) | Vision-aware self-learning design and hardware boundaries |
| [RELEASE.md](RELEASE.md) | v2.0.0 source-release runbook, validation commands, and artifact policy |
| [docs/inference-server.md](docs/inference-server.md) | Phase 27 server startup, endpoints, SDK setup, safety, auth, and smoke tests |
| [.github/release-notes/v2.0.0.md](.github/release-notes/v2.0.0.md) | GitHub Release body for v2.0.0 |

---

## Citation

If you use aarambh-ai in your research, please cite it as follows:

```bibtex
@software{aarambh_ai_2026,
  title        = {aarambh-ai: A Ground-Up LLM in Rust},
  author       = {Aarambh Dev Hub},
  year         = {2026},
  url          = {https://github.com/AarambhDevHub/aarambh-ai},
  version      = {2.0.0},
  license      = {Apache-2.0},
}
```

---

## Support

- Star the repo on [GitHub](https://github.com/AarambhDevHub/aarambh-ai)
- Open [issues](https://github.com/AarambhDevHub/aarambh-ai/issues) for reproducible bugs and clear feature requests
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
