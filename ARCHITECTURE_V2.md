# ARCHITECTURE_V2.md — aarambh-ai v2.0

> Companion to `ARCHITECTURE.md`. This document covers **only what v2.0
> adds or changes** — sections here are numbered to continue directly from
> v1's Section 18. Anything not mentioned here (tokenizer, RMSNorm, GQA,
> SwiGLU, quantisation, safety layer, etc.) is unchanged from v1.0.0.
> Read `ARCHITECTURE.md` first; this is the delta on top of it.

---

## Table of Contents

19. [What's New in v2.0](#19-whats-new-in-v20)
20. [Updated Workspace — 16 Library Crates](#20-updated-workspace--16-library-crates)
21. [Long Context: RoPE Scaling](#21-long-context-rope-scaling)
22. [Evaluation Harness](#22-evaluation-harness)
23. [DoRA — Weight-Decomposed Low-Rank Adaptation](#23-dora--weight-decomposed-low-rank-adaptation)
24. [Vision: Encoder, Projector, Fusion](#24-vision-encoder-projector-fusion)
25. [Vision-Language Training](#25-vision-language-training)
26. [Mixture of Experts](#26-mixture-of-experts)
27. [Multi-GPU Training](#27-multi-gpu-training)
28. [DPO — Direct Preference Optimisation](#28-dpo--direct-preference-optimisation)
29. [Speculative Decoding](#29-speculative-decoding)
30. [Tool Use / Function Calling](#30-tool-use--function-calling)
31. [Inference Server](#31-inference-server)
32. [Updated Dependency Layers](#32-updated-dependency-layers)
33. [Updated Memory & Compute Estimates](#33-updated-memory--compute-estimates)
34. [Updated Hardware Strategy](#34-updated-hardware-strategy)
35. [What's Explicitly Out of Scope](#35-whats-explicitly-out-of-scope)

---

## 19. What's New in v2.0

v1.0.0 shipped a complete, from-scratch LLM pipeline: pretraining, SFT,
LoRA/QLoRA, GRPO, INT4/GGUF quantisation, Flash Attention v2 CUDA kernels, a
thinking engine, safety guardrails, and CPU-capable self-learning. v2.0
extends that pipeline in three directions:

1. **Deeper on text** — longer context, a real evaluation harness, a
   stronger fine-tuning method (DoRA), a second alignment path (DPO), and
   faster inference (speculative decoding, tool calling).
2. **Wider — a new modality** — a frozen vision encoder + trainable
   projector gives the model the ability to see and reason about images,
   trained via free public datasets.
3. **Bigger training infra** — Mixture of Experts and multi-GPU data
   parallelism, both explicitly framed as systems-learning phases whose
   payoff is measured by the new evaluation harness rather than assumed.

**Still true in v2.0, unchanged from v1:** no pretrained checkpoints,
adapters, or GGUF files are released. This is a source/engineering release.
See `RELEASE.md`.

---

## 20. Updated Workspace — 16 Library Crates

Three new crates. Everything else is extended in place — no crate is
removed or renamed.

```
aarambh-ai/
├── Cargo.toml                        ← [workspace] manifest, shared dependencies
├── ARCHITECTURE.md / ARCHITECTURE_V2.md
├── ROADMAP.md / ROADMAP_V2.md
├── SELF_LEARNING.md / SELF_LEARNING_V2.md
│
├── crates/
│   │   ...Layers 0–5 from v1.0.0, extended (see sections 21–30 below)...
│   │
│   ├── aarambh-ai-eval/              ← NEW, LAYER 5: Evaluation harness
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── ppl.rs                ← perplexity-on-holdout
│   │       ├── harness.rs            ← EvalTask trait, run_all()
│   │       ├── tasks/
│   │       │   ├── mmlu_lite.rs
│   │       │   ├── hellaswag.rs
│   │       │   ├── gsm8k_subset.rs   ← reuses MathVerifier
│   │       │   ├── humaneval_lite.rs ← reuses CodeVerifier
│   │       │   └── vqa_subset.rs     ← added in Phase 20, vision QA task
│   │       └── report.rs             ← Scorecard, to_markdown(), to_json()
│   │
│   ├── aarambh-ai-vision/            ← NEW, LAYER 3: Vision encoder + projector
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── encoder.rs            ← ClipVisionEncoder (frozen ViT, SafeTensors)
│   │       ├── preprocess.rs         ← resize/crop/normalise (`image` crate)
│   │       ├── projector.rs          ← Projector MLP (trainable)
│   │       ├── fusion.rs             ← interleave_image_tokens()
│   │       └── instruct_data.rs      ← VqaExample, JSONL schema
│   │
│   └── aarambh-ai-serve/             ← NEW, LAYER 6: Inference server
│       └── src/
│           ├── lib.rs
│           ├── server.rs             ← routing, OpenAI-compatible endpoints
│           ├── batching.rs           ← ContinuousBatcher
│           └── session.rs            ← per-request KV cache + safety layer
│
└── aarambh-ai/                       ← LAYER 6: CLI binary
    └── src/cmd/
        ├── ...train.rs / infer.rs / finetune.rs / quantise.rs / convert.rs / eval.rs...
        └── serve.rs                  ← NEW: `aarambh-ai serve`
```

### Extended (not new) crates in v2.0

| Crate | v2.0 additions |
|---|---|
| `aarambh-ai-nn` | `rope_scaling.rs` (Phase 16), `moe.rs` + `dispatch.rs` (Phase 22) |
| `aarambh-ai-model` | `rope_scaling: Option<RopeScalingConfig>` and `moe: Option<MoeConfig>` on model config |
| `aarambh-ai-train` | `distributed.rs` (Phase 23), long-context continued-pretraining recipe, MoE aux loss |
| `aarambh-ai-finetune` | `dora.rs` (Phase 18), `vlm_dora.rs` (Phase 20), `dpo.rs` (Phase 24) |
| `aarambh-ai-inference` | `speculative.rs` (Phase 25), `grammar.rs` + `tool_calling.rs` (Phase 26), `--image` flag |
| `aarambh-ai-tokenizer` | `<image>` / `<image_end>` reserved special token strings, IDs, and validation |
| `aarambh-ai-selflearn` | `replay_buffer.rs` v2 schema, `vision_verifier.rs`, `gating.rs` — see `SELF_LEARNING_V2.md` |

### Updated Crate Count

```
v1.0.0: 14 crates (13 library + 1 binary)
v2.0.0: 16 library crates + 1 binary = 17 total
```

---

## 21. Long Context: RoPE Scaling

v1's RoPE (`ARCHITECTURE.md` §6.3) uses fixed rotation frequencies computed
once at `max_seq_len`. This works well up to the trained context length but
degrades sharply beyond it — the model has never seen rotation angles for
positions past what it trained on.

**YaRN** (Yet another RoPE extensioN) fixes this by interpolating between
the original and a scaled frequency, with a ramp function that leaves
high-frequency (local, short-range) dimensions mostly untouched and
interpolates low-frequency (long-range) dimensions more aggressively:

```
For each RoPE dimension pair i:
  if i is "high frequency" (< beta_fast threshold):  use original θᵢ unchanged
  if i is "low frequency"  (> beta_slow threshold):  use θᵢ / scale_factor
  in between:                                        ramp linearly between the two
```

The simpler **NTK-aware** alternative just rescales `rope_theta` itself:

```
θ_new = rope_theta × factor^(d / (d − 2))
```

NTK-aware is a one-line config change and a reasonable first thing to try;
YaRN needs a short continued-pretraining pass on long documents but
generalises better. Both are implemented; `RopeScalingConfig.method`
selects between them. `method: None` reproduces v1.0.0's RoPE exactly —
this is enforced by a regression test (`rope_scaling_none_matches_v1_output_exactly`).

Implementation notes:

- `RopeScalingConfig` is defined in `aarambh-ai-core` because `ModelConfig`
  owns the serialized schema. `aarambh-ai-nn::rope_scaling` owns the YaRN,
  NTK-aware, and linear frequency math.
- `AarambhModel` no longer stores a full `[max_seq_len, max_seq_len]` causal
  mask. Causal attention is passed to kernel dispatch directly, so CUDA
  FlashAttention can run with `causal = true` and CPU/Candle fallback only
  materializes the per-call mask it needs.
- Inference KV caches are created with `KVCache::with_capacity(max_seq_len)`
  so long-context autoregressive decoding writes into fixed cache storage
  instead of concatenating tensors on every token.
- `aarambh-ai-train` accepts an optional `context_schedule` for progressive
  loader rebuilds at 4K, 8K, and 16K while preserving model and optimizer state.

**Why continued pretraining, not from scratch:** the model already knows
how to use short-range RoPE angles. Continued pretraining on long documents
at progressively increasing context (4K → 8K → 16K) teaches it to make use
of the newly-available long-range positions without re-learning everything
else.

---

## 22. Evaluation Harness

v1.0.0 measured quality with perplexity alone (`ARCHITECTURE.md` §9.7's
training output logs `loss` and implicitly PPL via `eval_steps`). That's a
single, fairly blunt signal — it doesn't tell you if the model can actually
answer a multiple-choice question, solve a word problem, or write working
code.

`aarambh-ai-eval` adds four free, public benchmark subsets on top of PPL:

| Task | What it measures | Scoring |
|---|---|---|
| MMLU-lite | General knowledge, multiple choice | Next-token logprob comparison over A/B/C/D |
| HellaSwag | Commonsense sentence completion | Logprob ranking over candidate completions |
| GSM8K-subset | Grade-school math word problems | Exact-match, reuses `MathVerifier` from `aarambh-ai-finetune` |
| HumanEval-lite | Python code generation | pass@1 via sandboxed execution, reuses `CodeVerifier` |

Each task implements a shared `EvalTask` trait so new tasks (like Phase
20's `vqa_subset`) plug in without touching the harness runner itself.

```rust
trait EvalTask {
    fn name(&self) -> &'static str;
    fn run(&self, context: &EvalContext, config: &EvalConfig) -> TaskScore;
}
```

The CLI entrypoint is `aarambh-ai eval`. It loads the same TOML config,
tokenizer, SafeTensors/GGUF checkpoint, device, and dtype plumbing used by
training and inference, then writes a JSON scorecard and/or Markdown table.
`--compare before.json after.json` is pure scorecard comparison and does not
load a model. HumanEval-lite is intentionally opt-in with `--allow-code-exec`
because it executes generated Python tests through `CodeVerifier`.

**Design principle:** every phase from 16 onward that changes model
behaviour reports a before/after `Scorecard` (`aarambh-ai eval --compare
before.json after.json`). Claims about whether a change helped are measured,
not assumed — see Phase 22 (MoE) and Phase 24 (DPO) in `ROADMAP_V2.md`,
both of which are explicitly gated on what the scorecard shows rather than
treated as automatic wins.

---

## 23. DoRA — Weight-Decomposed Low-Rank Adaptation

v1's LoRA (`ARCHITECTURE.md` §12.1) adapts a frozen weight `W₀` by adding a
low-rank update: `W = W₀ + BA` (scaled by `α/r`). This changes both the
*magnitude* and *direction* of each weight vector together, coupled by the
same low-rank update.

DoRA decomposes `W` into magnitude and direction first:

```
W₀ = m · (V / ||V||_c)        where m = column-wise magnitude, V/||V||_c = unit direction

DoRA update:
  direction' = (V + BA) / ||V + BA||_c     ← LoRA-style low-rank update, but only to direction
  m'         = trainable magnitude vector   ← trained independently, full rank
W'         = m' · direction'
```

In this codebase, Candle linear weights are stored as `[out_dim, in_dim]`,
so the implementation computes the DoRA magnitude and norm per output row.

This decoupling is the whole point: magnitude and direction can now move
independently, which published results show consistently outperforms
plain LoRA at the same rank, at a small additional parameter cost (one
magnitude vector per adapted layer).

`DoraLinear` mirrors `LoraLayer`'s structure closely — same
`target_modules` matching (`attn.wq/wk/wv/wo`, FFN gate/up/down), same
adapter save/merge pattern, same `QDoRA` variant pairing a frozen INT4 base
with trainable BF16 adapter + magnitude. It is a genuine drop-in alternative,
not a replacement — `aarambh-ai finetune sft`, `qlora`, `dora`, and `qdora`
all remain available, and Phase 18's milestone is an honest
side-by-side comparison via the eval harness rather than an assumed win.

DoRA/QDoRA's memory profile is essentially identical to LoRA/QLoRA's (one
extra magnitude vector is negligible next to the base weights), so it
remains i3-capable at small scale, same as v1's QLoRA.

---

## 24. Vision: Encoder, Projector, Fusion

### Design choice: frozen encoder, trainable projector

Training a vision encoder from scratch needs dataset scale and compute far
beyond free Kaggle sessions. v2.0 follows the LLaVA-style approach instead:
a **frozen**, pretrained ViT-class encoder (CLIP-B/32 scale, ~86M params,
loaded from public SafeTensors weights via Candle) turns an image into a
grid of patch embeddings. A small **trainable** projector MLP maps those
embeddings into the language model's `d_model` space, so each image patch
becomes something that looks, to the rest of the model, like a token
embedding.

```
Image (H×W×3)
   │
   ▼
Patchify + linear patch embed (frozen) → N patches × vit_d_model
   │
   ▼
ViT transformer blocks (frozen, pretrained CLIP weights)
   │
   ▼
Projector MLP (trainable): vit_d_model → hidden → llm_d_model
   │
   ▼
N "image tokens" in llm_d_model space
   │
   ▼
Spliced into the text token sequence at the <image> special token position
   │
   ▼
...rest of aarambh-ai's existing decoder, completely unmodified from v1...
```

### Why this stays inside the Dependency Policy

CLIP weights are loaded as SafeTensors directly through `candle-core` — the
same loading path `aarambh-ai-weights` already uses for the language model
(`ARCHITECTURE.md` §4, `aarambh-ai-weights/src/safetensors.rs`). No
PyTorch bindings (`tch-rs`), no ONNX Runtime, no Python FFI. The only new
dependency is the `image` crate, used purely for local decode/resize/
normalise — no network calls, no external service.

Implementation note: Phase 19 accepts public HuggingFace CLIP-B/32
SafeTensors and normalizes their tensor names into Aarambh's canonical
encoder layout at load time, including reshaping the CLIP conv patch weight
into the linear patchifier matrix.

### Fusion: why prefix-splice, not cross-attention

v2.0 uses LLaVA-style **prefix fusion** (image tokens spliced directly into
the input sequence) rather than a separate cross-attention mechanism
(Flamingo-style). This is deliberately the simpler option: it requires zero
changes to `TransformerBlock`, `GroupedQueryAttention`, or the KV cache —
image tokens are just tokens as far as the decoder is concerned. The
trade-off is that image tokens consume context budget alongside text
tokens, which is part of why long context (Phase 16 / §21) is sequenced
before vision — more headroom for image + text together.

### Two-stage training (see §25)

The encoder never trains. The projector trains in two stages: first alone
(frozen LLM, frozen encoder — Phase 19), then alongside a DoRA-adapted LLM
for instruction-following (Phase 20). This mirrors LLaVA's own two-stage
recipe and keeps each stage cheap enough for Kaggle's free quota.

---

## 25. Vision-Language Training

Phase 19 (§24) produces a projector that can turn images into "readable"
tokens, trained only on captioning — descriptive, not conversational.
Phase 20 extends this to open-ended visual question answering.

**What trains, what doesn't:**

| Component | Phase 19 (projector pretrain) | Phase 20 (VQA instruction tuning) |
|---|---|---|
| Vision encoder | frozen | frozen |
| Projector | **trainable** | frozen or trainable (config flag) |
| LLM attention/FFN | frozen | **DoRA-adapted** (§23) |

Using DoRA rather than full fine-tuning for the Phase 20 LLM side is a
direct cost decision: full fine-tuning of even the Small scale on
image-text pairs would be well outside free Kaggle quota; DoRA adapters
keep the trainable parameter count small enough to fit.

**Loss masking** follows exactly the same pattern as v1's SFT
(`ARCHITECTURE.md` §12.3): image tokens and question tokens are masked out
of the loss, only answer (and, if present, thinking) tokens carry gradient.
`build_loss_mask` from Phase 9 is reused unmodified — VQA examples just
have image tokens prepended to what SFT already treats as prompt context.

**Composability with the thinking engine:** because fusion happens before
the decoder (§24), the existing `ThinkingController`
(`ARCHITECTURE.md` §7.4) needs no changes. A `<think>` block generated after
image tokens behaves identically to one generated after text-only context —
the budget and mode logic don't know or care that some of the preceding
context came from an image.

---

## 26. Mixture of Experts

### The honest framing

MoE's value proposition is routing to *specialized* experts across a
*large* total parameter count while keeping *active* compute per token
low. At Tiny/Small scale (25M–117M dense-equivalent), there isn't much
room for experts to meaningfully specialize — this phase is scoped and
documented as a systems-engineering exercise first, a quality improvement
second, with the eval harness (§22) deciding which framing turned out to
be true for a given run.

### Mechanism

```
MoeConfig { num_experts, top_k, expert_ffn_dim, aux_loss_weight }

Router: linear gate, d_model → num_experts logits per token
top_k_gating(logits) → indices + softmax weights, over the top_k experts only

MoeFfn: num_experts independent SwiGLU FFNs (same structure as v1's
        single dense FFN, §6.6), each token dispatched to its top_k experts
        and weighted-summed

Load-balancing auxiliary loss: standard switch-transformer-style loss that
        penalises uneven expert utilisation, added to the main
        cross-entropy loss with weight aux_loss_weight
```

`MoeFfn` is a **drop-in replacement** at the block level for the dense
`SwiGluFfn` — same input/output shape — selected per-layer or every-N-layers
via `moe: Option<MoeConfig>` on the model config. `moe: None` reproduces
v1's dense FFN exactly.

### Dispatch implementation

v2.0 ships a dense masked-matmul dispatch (every expert computes on every
token, then masks/weights by the router) as the first, simplest-correct
implementation. A sparse/grouped dispatch (only compute the selected
experts per token) is called out as an optional follow-up in
`ROADMAP_V2.md`, not a Phase 22 requirement — correctness first, efficiency
second.

---

## 27. Multi-GPU Training

Kaggle occasionally grants 2×T4 sessions instead of the usual single-GPU
session. v2.0 adds data-parallel training via Candle's NCCL-based collective
operations, with single-GPU as the default and unconditional fallback.

```
DistributedConfig { world_size, rank, backend: Nccl }

Per training step:
  1. Each rank forward+backward passes on its own data shard
  2. all_reduce_gradients() averages gradients across ranks
  3. Optimiser step proceeds identically to single-GPU (§9.4)
  4. Checkpoint save only from rank 0
```

`Trainer` gains an `Option<DistributedConfig>` field. When `None`
(the default, and the only path on the i3 or a single-GPU Kaggle session),
behaviour is byte-identical to v1.0.0's training loop — this is enforced by
a regression test.

---

## 28. DPO — Direct Preference Optimisation

v1's alignment path is GRPO (`ARCHITECTURE.md` §12.4) — group-sampled
completions, scored by a **deterministic verifier**, advantage-weighted
policy update with a KL penalty. This works well when correctness is
checkable (math, code, format) but doesn't have an obvious equivalent for
open-ended chat quality, where "which response is better" is a preference
judgment, not a pass/fail check.

DPO trains directly on `(prompt, chosen, rejected)` preference pairs,
without needing a separate reward model or an RL sampling loop:

```
dpo_loss = -log σ(β × [(log π(chosen|prompt) − log π_ref(chosen|prompt))
                      − (log π(rejected|prompt) − log π_ref(rejected|prompt))])
```

where `π` is the model being trained and `π_ref` is a frozen reference
checkpoint (typically the pre-DPO model). A `reference_free` variant
exists for when keeping a second frozen copy in memory isn't affordable.

**Division of labour going forward:** GRPO (verifier-based) stays the
preferred path for math/code/format tasks where correctness is checkable.
DPO (preference-based) is the new preferred path for open-ended chat
quality, where "better" is a judgment call rather than a checkable fact.
Both remain available; they are complementary, not competing.

Like DoRA fine-tuning, `DpoTrainer` can wrap a DoRA adapter instead of doing
a full fine-tune, keeping it affordable on a single Kaggle T4.

---

## 29. Speculative Decoding

Uses the existing Tiny (25M) and Large (1.3B) scales together at inference
time, no new model needed:

```
Draft model (Tiny):  proposes K tokens ahead, cheaply
Target model (Large): verifies all K tokens in a single forward pass

Accept the longest correct prefix.
Resample the first rejected token from the target model's own (corrected)
distribution — this is what makes speculative decoding exact rather than
an approximation: the final output distribution is statistically
identical to decoding with the target model alone, just faster.
```

This requires draft and target to share a tokenizer/vocabulary — both
already use the same `aarambh-ai-tokenizer` BPE vocab (`ARCHITECTURE.md`
§6.1), so no new tokenizer work is needed. A mismatched-vocab guard errors
clearly rather than silently producing garbage if someone points it at
incompatible checkpoints.

Speculative decoding is validated two ways: wall-clock speedup (target
≥1.8× tokens/second) and, more importantly, an eval-harness (§22) check
that the *output distribution* is unchanged — confirming the speedup isn't
coming from a quality regression.

---

## 30. Tool Use / Function Calling

### Grammar-constrained decoding

SFT-only tool calling (train the model to usually emit valid JSON) is not
enough — "usually valid" JSON breaks downstream parsers. v2.0 adds a
grammar layer that constrains sampling itself: a `JsonSchemaGrammar`
compiles a JSON Schema into a per-step valid-next-token mask, applied
during the existing sampling step (`ARCHITECTURE.md` §8.2) so the model
is *structurally incapable* of emitting invalid JSON when inside a tool-call
span, rather than merely trained to usually avoid it.

```
ToolDefinition { name, description, parameters: JsonSchema }

Decode loop (unchanged KV cache / prefill+decode from §8.1):
  ...normal token-by-token generation...
  on entering a tool_call span:
    sampler restricted to grammar's currently-valid next tokens
  on exiting the span:
    sampler reverts to normal (Top-K/Top-P/Temperature, §8.2)
```

### Composability with thinking

`ToolCallController` wraps the existing `ThinkingController`
(`ARCHITECTURE.md` §7.4) rather than replacing it: the model may think
first (bounded by its existing budget), then either answer directly or
emit a grammar-constrained tool call. Neither system needs to know about
the other beyond this wrapping.

---

## 31. Inference Server

### Local-only, source-only

`aarambh-ai-serve` exposes an OpenAI-compatible HTTP/SSE API
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`) over your own
locally-loaded checkpoints. This is a **self-hosted, local server** — it
does not publish weights, and `/v1/models` only ever lists checkpoints
already present on the machine it's running on. Same "source, not
artifacts" policy as the rest of v2.0 (see `RELEASE.md`).

### Continuous batching

Rather than waiting for a fixed batch of requests before running a forward
pass (static batching), `ContinuousBatcher` merges concurrent decode steps
dynamically — a new request can join at the next available step instead of
waiting for the current batch to fully finish. Each request keeps its own
KV cache (`ARCHITECTURE.md` §8.1); the batcher only shares the forward-pass
call, not the cache state, so concurrent requests cannot cross-contaminate
each other's context.

### What's reused unmodified

The safety layer (`ARCHITECTURE.md` §13) applies to every server request by
default, same as CLI `infer`. Sampling, thinking, and (if configured) tool
calling all pass through unchanged — the server is a new transport around
the existing inference engine, not a new inference path.

---

## 32. Updated Dependency Layers

```
Layer 0  aarambh-ai-core
Layer 1  aarambh-ai-tokenizer   aarambh-ai-data
Layer 2  aarambh-ai-nn          aarambh-ai-kernel
Layer 3  aarambh-ai-model       aarambh-ai-weights    aarambh-ai-quant     aarambh-ai-vision
Layer 4  aarambh-ai-train       aarambh-ai-finetune
Layer 5  aarambh-ai-inference   aarambh-ai-safety     aarambh-ai-selflearn  aarambh-ai-eval
Layer 6  aarambh-ai-serve       aarambh-ai (binary)
```

`aarambh-ai-vision` sits at Layer 3 (same layer as `aarambh-ai-model`) since
it produces embeddings consumed by the model assembly layer, not the raw
neural-net primitives layer. `aarambh-ai-eval` sits at Layer 5 since it
needs a fully-assembled, fine-tunable model to evaluate. `aarambh-ai-serve`
sits at Layer 6 alongside the CLI binary — it's a second, HTTP-shaped
entry point into the same Layer 5 inference/safety stack the CLI uses.

The same rule as v1 applies unchanged: every crate may only depend on
crates in the same or lower layer, enforced by `Cargo.toml`.

### New Dependency Policy Entries

| Dependency | Allowed crates | Reason |
|---|---|---|
| `image` | `aarambh-ai-vision` | Local decode/resize/normalise only — no network calls |
| NCCL bindings (via `candle-core` CUDA features) | `aarambh-ai-train` | Multi-GPU collective ops (§27) |
| HTTP/SSE server framework | `aarambh-ai-serve` | Request routing, streaming |

**Still forbidden everywhere, unchanged from v1:** PyTorch bindings
(`tch-rs`), ONNX Runtime (`ort`), Python FFI, `llama.cpp` as a backend.
Pretrained CLIP weights are loaded as SafeTensors through `candle-core`,
never as a PyTorch checkpoint requiring `tch-rs`.

---

## 33. Updated Memory & Compute Estimates

### Vision Addition (on top of v1's per-scale table, `ARCHITECTURE.md` §17)

| Component | Params | F32 Memory |
|---|---|---|
| CLIP-B/32 vision encoder (frozen) | ~86M | ~344 MB |
| Projector MLP (trainable) | ~2–4M (depends on `hidden_mult`) | ~8–16 MB |

Frozen encoder weights need no gradient or optimiser state — only forward
activations, same as any frozen component. The trainable projector's
AdamW state follows the same 4×-weights rule as v1 (`ARCHITECTURE.md` §17).

### DoRA vs LoRA Memory (per adapted layer)

| Method | Trainable params per layer |
|---|---|
| LoRA (v1) | `2 × r × d_model` (A and B matrices) |
| DoRA (v2) | `2 × r × d_model` (A and B) `+ d_model` (magnitude vector) |

The magnitude vector adds a negligible fraction on top of LoRA's footprint
— DoRA remains i3-capable at the same scales LoRA was.

### MoE Memory (Small-MoE example, `num_experts=8, top_k=2`)

| | Dense Small (v1) | Small-MoE (v2) |
|---|---|---|
| Active params/token | 117M | ~117M (top_k=2 of 8 experts, ≈ dense-equivalent active compute) |
| Total params | 117M | ~117M × (expert_ffn_dim scaling factor) — total grows with `num_experts`, active compute does not |

The whole point of top-k routing: total parameter count (and total weight
memory) grows with `num_experts`, but active compute per token stays close
to the dense-equivalent — this is the trade MoE is making, and Phase 22's
milestone is checking whether that trade pays off in the eval harness at
this scale.

---

## 34. Updated Hardware Strategy

### Your Local Machine (i3-1115G4, 8 GB RAM, Pop OS)

Everything from v1's i3 capability list (`ARCHITECTURE.md` §18) still
applies unchanged. v2.0 adds:

- Eval harness on Tiny/Small (PPL + MMLU-lite/HellaSwag subsets — small
  enough to run locally; GSM8K/HumanEval-lite's execution sandboxing is
  also i3-capable at small scale)
- DoRA/QDoRA fine-tuning of Small (same ~400 MB peak class as v1's QLoRA)
- Grammar-constrained tool-call inference (decoding-time cost, not
  training-time — i3-capable)
- Inference server for Tiny/Small checkpoints (CPU serving, no GPU needed)

**Explicitly NOT i3-capable in v2.0:**
- Vision encoder forward passes (frozen ViT adds real CPU cost per turn —
  see `SELF_LEARNING_V2.md` for the specific gating decision)
- Any MoE training (dispatch overhead assumes GPU-scale batch sizes)
- Multi-GPU training (obviously — needs 2 GPUs)
- Full VLM instruction-tuning runs (Phase 20) — projector *inference* on a
  single image is fine on i3; *training* the VLM stack is Kaggle-scoped

### Kaggle GPU (unchanged tiers, new workloads)

| Scale/Workload | GPU | Notes |
|---|---|---|
| Long-context continued pretraining | T4/P100 | Progressive 4K→8K→16K warm-up |
| Vision projector pretrain + VQA tuning | T4/P100 | Frozen encoder, small trainable surface |
| MoE training | P100/A100 recommended | Dispatch overhead benefits from more VRAM headroom |
| Multi-GPU (2×T4) | 2×T4 | Not always available — single-GPU fallback is the default path |
| DPO | T4 | DoRA-wrapped, same cost class as Phase 18 |
| Speculative decoding validation | T4 | Inference-time only, no training |

---

## 35. What's Explicitly Out of Scope

Carried over from `ROADMAP_V2.md` for architectural completeness:

- Releasing any pretrained checkpoint, adapter, or GGUF file
- Multi-node distributed training (2-GPU single-node only)
- Public/hosted deployment of the inference server (local-only)
- Video or audio modalities (vision is images only, via a frozen ViT)
- Synthetic API-generated training data (all v2.0 datasets are free and
  public; paid synthetic data remains a future upgrade, not a v2.0
  requirement)
- Sparse/grouped MoE dispatch (v2.0 ships the simpler dense-masked-matmul
  version; sparse dispatch is a documented future optimisation)

These are natural v3 candidates once budget and a released model exist.
