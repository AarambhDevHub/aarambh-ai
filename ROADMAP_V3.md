# ROADMAP_V3.md — aarambh-studio v3.0

> From first principles. From zero. From Rust.
>
> Step-by-step build plan for v3.0. Every phase ends with working, testable
> code. Builds on the completed v2.0.0 base (Phases 0–28, all ✅). No
> pretrained checkpoints are released as part of v3.0 — this is a
> source/engineering release, same policy as v1.0.0 and v2.0.0.

---

## How to Read This Roadmap

Each phase has:
- **Goal** — exactly what you will have when this phase is done
- **Tasks** — the checklist to follow, in order, grouped by crate
- **Tests** — what you write to prove it works
- **Milestone** — how you know you are done, with the git tag to cut

Work top to bottom. Do not skip phases — each phase depends on the ones
before it. Phases 29–32 are architecture-core changes and are deliberately
sequenced first because everything after them (MoE, distillation, QAT)
trains *on top of* the new attention stack, not the old one. Phases 33–36
are training/efficiency phases. Phases 35–36 are multimodal. Phase 37 is
agentic. Phase 38 is memory/self-learning. Phase 39 adds the Max thinking
mode. Phase 40 is the crates.io publish.

---

## Phase Map (Quick Reference)

```
Phase 29 →  Gated DeltaNet (hybrid linear attention)   (10–14 days)  [Kaggle] ✅
Phase 30 →  DeepSeek Sparse Attention (DSA)             (10–14 days)  [Kaggle] ✅
Phase 31 →  DeepSeek-style fine-grained MoE + shared    (10–14 days)  [Kaggle] ✅
            expert routing (v3 upgrade of v2 dense MoE)
Phase 32 →  Multi-Token Prediction (MTP)                (7–10 days)   [Kaggle] ✅
Phase 33 →  On-policy distillation                      (10–14 days)  [Kaggle] ✅
Phase 34 →  Native QAT (quantization-aware training)    (7–10 days)   [i3 + Kaggle] ✅
Phase 35 →  Native video understanding                  (14–18 days)  [Kaggle] ✅
Phase 36 →  Native document understanding               (10–14 days)  [Kaggle] ✅
Phase 37 →  Long-horizon tool-use chains                (10–14 days)  [i3 + Kaggle] ✅
Phase 38 →  Forgetting diagnostics tied to Manas         (7–10 days)   [i3 + Kaggle] ✅
Phase 39 →  Max thinking mode (5th reasoning depth)      (5–7 days)    [i3 + Kaggle]
Phase 40 →  crates.io publish (v3.0.0)                   (5–7 days)    [all]
```

**Total realistic estimate: 105–145 days (~3.5–4.8 months)**

---

## Why This Order

1. **29–30 (Gated DeltaNet, DSA) come first** because they replace the
   attention primitive that every later phase trains on top of. Doing
   attention surgery *after* MoE or distillation is built would mean
   re-validating both against a moving target. Get the attention stack
   stable and eval-harness-verified (`aarambh-studio-eval`, v2 Phase 17) before
   anything else touches it.
2. **31 (fine-grained MoE)** comes right after because it is the other
   half of the "efficient frontier architecture" story, and because v2's
   Phase 22 already shipped a dense-masked-matmul MoE — this phase is an
   upgrade path on an existing, tested crate, not a fresh build. It
   benefits from having the new attention stack (29–30) already in place
   so routing is tuned against the real v3 forward pass, not the old one.
2. **32 (MTP)** follows MoE directly because multi-token prediction heads
   are cheapest to validate once the router is stable, and MTP's auxiliary
   loss is reused as a training signal by both Phase 33 (distillation) and
   the base pretraining recipe going forward.
3. **33–34 (distillation, QAT)** are training-efficiency phases that make
   sense once the target architecture (attention + MoE + MTP) is frozen —
   distilling or quantizing a model whose architecture is still moving is
   wasted work.
4. **35–36 (video, document understanding)** come together because they
   share a vision encoder and multimodal fusion path (both build on v2's
   Phase 19 `aarambh-studio-vision` crate). Video first because it is the
   larger architectural lift (temporal fusion); document understanding
   reuses the same patch encoder with a layout-aware head, so it is
   naturally the smaller follow-on phase.
5. **37 (long-horizon tool use)** comes after multimodal because real
   agent chains increasingly need to reason over tool outputs that include
   images and documents (a search result page, a screenshot, a PDF
   attachment) — this phase upgrades v2 Phase 26's single-call, emit-only
   tool use into multi-step chains, and benefits from 35–36 being done
   first so chains can include multimodal tool results.
6. **38 (forgetting diagnostics)** comes late deliberately — it needs a
   model with enough moving parts (new attention, MoE, distillation, new
   modalities) to actually have something worth diagnosing. This phase
   also directly informs Manas's own anti-forgetting design, hence "tied
   to Manas" — see `SELF_LEARNING_V3.md`.
7. **39 (Max thinking mode)** is placed near the end so it inherits every
   v3 architecture change — a longer thinking budget is only worth adding
   once the model underneath it (new attention stack, MoE, MTP, QAT) is
   the version that will actually use those extra tokens well, not the
   pre-v3 architecture.
8. **40 (crates.io)** is last, exactly like v1's Phase 15 and v2's Phase 28
   discipline: ship the *code* as source once it is proven, never ship
   unproven code or unreleased weights.

---

## Workspace `Cargo.toml` Additions

```toml
[workspace]
members = [
    # ...existing v1.0.0 + v2.0.0 members unchanged...
    "crates/aarambh-studio-core",
    "crates/aarambh-studio-tokenizer",
    "crates/aarambh-studio-data",
    "crates/aarambh-studio-nn",
    "crates/aarambh-studio-kernel",
    "crates/aarambh-studio-model",
    "crates/aarambh-studio-weights",
    "crates/aarambh-studio-quant",
    "crates/aarambh-studio-train",
    "crates/aarambh-studio-finetune",
    "crates/aarambh-studio-inference",
    "crates/aarambh-studio-safety",
    "crates/aarambh-studio-selflearn",
    "crates/aarambh-studio-eval",
    "crates/aarambh-studio-vision",
    "crates/aarambh-studio-serve",

    # new in v3.0
    "crates/aarambh-studio-distill",     # Phase 33
    "crates/aarambh-studio-agent",       # Phase 37

    "aarambh-studio",
]
```

Two new crates (`aarambh-studio-distill`, `aarambh-studio-agent`). Everything else
extends existing crates, most heavily `aarambh-studio-nn` (attention, MoE, MTP),
`aarambh-studio-vision` (video, documents), and `aarambh-studio-quant` (native QAT).
No new external dependencies beyond what's listed in each phase's
Dependency Policy note.

---

## Phase 29 — Gated DeltaNet (Hybrid Linear Attention)

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

### Goal
A hybrid attention block that interleaves Gated DeltaNet linear-attention
layers with a minority of full-attention layers, retrofitted onto an
existing pretrained checkpoint via continued pretraining rather than a
from-scratch rebuild — matching how 2026 open-weight labs (Qwen3-Next and
successors) describe adopting this mechanism as a "low-commitment"
modification slotted in during fine-tuning.

### Tasks

**`aarambh-studio-nn`:**
```
[x] src/gated_deltanet.rs
      Exact decayed delta-rule recurrence with causal depthwise q/k/v
      convolution, normalized q/k features, learnable alpha/beta gates,
      output RMSNorm and SiLU output gating
      Differentiable chunk-bounded Candle training/prefill form and optimized
      sequential CPU/CUDA recurrent decode form
      DeltaNetState stores the fixed recurrent matrix and bounded convolution
      history; no sequence-length-dependent KV allocation

[x] src/block.rs + src/kvcache.rs
      TokenMixer enum: Attention | GatedDelta
      HybridKvCache enum: Full(KVCache) | Linear(DeltaNetState)
      Full-attention layers retain the existing GQA + RoPE/YaRN path

[x] aarambh-studio-core/src/config.rs
      AttentionKind: Full | GatedDeltaNet
      HybridAttentionSchedule selects one full-attention layer per configurable N
      Full-attention layers keep the existing GQA + RoPE/YaRN path
      (`ARCHITECTURE.md` §6.3, `ARCHITECTURE_V2.md` §21) completely
      unchanged — hybrid means "some layers differ," not "attention is
      replaced everywhere"
```

**`aarambh-studio-model`:**
```
[x] Model config gains `attention_schedule: Option<HybridAttentionSchedule>`
[x] Backward compatible: attention_schedule = None reproduces exact v1/v2
    all-full-attention behaviour
[x] Hybrid cache construction, capture, batched decode and tensor lookup
[x] New hybrid Medium/Large and CPU/CUDA smoke configs
      configs/wikitext103_medium_hybrid.toml
      configs/wikitext103_large_hybrid.toml
      configs/gated_deltanet_smoke.toml
      configs/wikitext103_hybrid_cuda_smoke.toml
```

**`aarambh-studio-train`:**
```
[x] Continued-pretraining recipe: load an existing v2 checkpoint, replace
    the scheduled layers' weights with freshly-initialised GatedDeltaNet
    parameters, keep untouched layers' weights loaded as-is, train at a
    reduced learning rate so the untouched layers do not drift far from
    their pretrained state while the new layers learn
[x] Dense and hybrid training use differentiable RMSNorm and verify gradients
    reach the final block and Gated DeltaNet projection parameters
[x] Retrofit validation command: eval-harness score (v2 Phase 17) before and after
    retrofit, on the same holdout, must not regress beyond a documented
    tolerance band before a trained retrofit is accepted
```

**`aarambh-studio-weights`:**
```
[x] Partial-checkpoint loading: load full-attention layer weights from an
    existing SafeTensors checkpoint while initialising GatedDeltaNet layer
    weights fresh, in a single load call
[x] GGUF keeps recurrent scalars and convolution weights full precision while
    quantising eligible rank-2 projections
```

**Cross-cutting integration:**
```
[x] Exact speculative-decoding rollback via hybrid cache snapshots/replay
[x] LoRA/QLoRA/DoRA/QDoRA projection targets and adapter merge support
[x] Calibration capture, inference sessions, continuous batching and serving
[x] CPU-parallel and CUDA recurrent kernels with portable fallback dispatch
[x] Associative-recall task and long-context dense-vs-hybrid benchmark script
```

### Data Setup

```bash
# Reuses the same continued-pretraining corpus style as v2 Phase 16
# (long-document data), since Gated DeltaNet's main payoff is long-context
# efficiency — retrofit on documents long enough to exercise it.
scripts/phase29_prepare_hybrid_retrofit.sh data
```

### Tests

```rust
#[test]
fn attention_schedule_none_matches_v2_full_attention_exactly() {
    // No hybrid schedule configured -> bit-identical to pre-v3 output.
}

#[test]
fn gated_deltanet_chunk_parallel_matches_sequential_recurrence() {
    // Training-time chunked form and inference-time recurrent form must
    // produce numerically equivalent outputs (within float tolerance).
}

#[test]
fn deltanet_state_size_is_constant_regardless_of_sequence_length() {
    // Unlike KV cache, linear-attention state must not grow with tokens
    // generated — this is the whole point of the mechanism.
}

#[test]
fn partial_checkpoint_load_preserves_full_attention_layer_weights_exactly() {}

// Hardware acceptance after continued training:
// compare dense and retrofit scorecards with `aarambh-studio eval compare`.
```

### Milestone
```
Hybrid Medium/Large configs and the complete retrofit code path are available
from an existing v2 checkpoint. A user-run continued-pretraining job must still
demonstrate eval scores within the documented tolerance band and measure 16K+
throughput on the target GPU before that specific trained checkpoint is
accepted; no pretrained checkpoint is distributed by this repository.

git commit -m "feat: Phase 29 — Gated DeltaNet hybrid linear attention"
git tag v3.0.0-alpha.1
```

---

## Phase 30 — DeepSeek Sparse Attention (DSA)

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

### Goal
A sparse-attention mechanism for the remaining full-attention layers
(those not converted to Gated DeltaNet in Phase 29) that reduces attention
compute and K/V memory bandwidth at long context by having each query attend
to a learned/scored subset of blocks rather than the full causal history.
Full K/V storage remains O(context); DSA reduces the selected working set,
not the asymptotic cache capacity.

### Tasks

**`aarambh-studio-nn`:**
```
[x] src/sparse_attention.rs
      DsaConfig — top_k (how many key blocks each query attends to),
      block_size (granularity of key selection, coarser than per-token)
      lightning_indexer() — a small auxiliary scoring path that ranks key
      blocks per query cheaply, before the expensive full attention matmul
      runs only over the selected blocks
      DsaAttention::forward() — two-stage: (1) index/score, (2) sparse
      gather + standard scaled-dot-product attention over selected blocks
      Falls back to full attention automatically when sequence length is
      below a configurable threshold (sparsity has no payoff on short
      sequences, only adds overhead)

[x] src/attention.rs
      AttentionKind gains a third variant: Full | GatedDeltaNet | Sparse
      HybridAttentionSchedule extended to place Sparse on the "full
      attention" layer slots from Phase 29's schedule
```

**`aarambh-studio-model`:**
```
[x] Model config gains `dsa_config: Option<DsaConfig>`, applies to
    whichever layers HybridAttentionSchedule marks as Sparse
[x] New configs layering DSA on top of Phase 29's hybrid schedules
      configs/medium_hybrid_dsa.toml
      configs/large_hybrid_dsa.toml
```

**`aarambh-studio-train`:**
```
[x] Indexer training: the lightning indexer needs its own small
    supervised signal (distilled from full-attention scores on a subset
    of training steps) so it learns to rank the same key blocks a full
    attention pass would have weighted highly
[x] K/V working-set benchmark harness: reports stored cache separately
    from selected K/V bytes at 4K/16K/32K context with and without DSA
```

### Tests

```rust
#[test]
fn dsa_falls_back_to_full_attention_below_length_threshold() {}

#[test]
fn lightning_indexer_selected_blocks_overlap_full_attention_top_blocks() {
    // Statistical check: indexer's top-ranked blocks agree with a full
    // attention pass's actual top-weighted blocks above a set threshold,
    // on a held-out sample.
}

#[test]
fn sparse_attention_output_shape_matches_full_attention_output_shape() {}

#[test]
fn selected_kv_working_set_at_32k_is_bounded_by_top_k_blocks() {}
```

### Milestone
```
DSA-enabled configs show measured attention working-set and bandwidth
reduction at 16K/32K context versus the Phase 29 hybrid-only baseline, while
reporting total stored K/V separately, with eval-harness scores
within documented tolerance. Combined with Phase 29, the attention stack
for v3.0 is now: mostly Gated DeltaNet, a minority of DSA-sparse layers,
zero fully-dense full-attention layers by default (dense full attention
remains available as a config fallback for debugging/comparison).

git commit -m "feat: Phase 30 — DeepSeek Sparse Attention (DSA)"
git tag v3.0.0-alpha.2
```

---

## Phase 31 — DeepSeek-Style Fine-Grained MoE + Shared Expert

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

### Goal
Upgrade v2 Phase 22's dense-masked-matmul MoE into a fine-grained routing
design: many small experts (rather than few large ones) plus one or more
always-active "shared" experts that every token passes through regardless
of router decisions — the pattern MiniMax and DeepSeek-derived
architectures use to reach near-frontier scores at low active-parameter
counts.

### Tasks

**`aarambh-studio-nn`:**
```
[x] src/moe.rs (extends v2's implementation)
      MoeConfig gains: num_shared_experts (default 0), fine_grained_factor
      (splits each v2-era expert into fine_grained_factor smaller experts
      of proportionally smaller FFN width, keeping total FFN capacity
      roughly constant while increasing expert count and specialisation)
      SharedExpertPath — always-active FFN, output added unconditionally
      to every token's routed-expert output before the residual connection
      Router unchanged in kind (top-k softmax gate) but now scores a
      larger, finer-grained expert pool

[x] Dense dispatch contract (extends v2's implementation)
      Dense-masked-matmul dispatch (v2's shipped approach) remains the
      default for Kaggle-scale training — sparse/grouped dispatch stays
      documented out-of-scope per v2 §35, still true in v3 unless
      hardware access changes
      Load-balancing auxiliary loss extended to account for the shared
      expert's fixed activation (it is excluded from the balancing loss
      since it is never routed, only always-on)
```

**`aarambh-studio-model`:**
```
[x] Model config's existing `moe: Option<MoeConfig>` (v2 §20) gains the
    new fields above; old MoeConfig values remain valid (fine_grained_factor
    defaults to 1, num_shared_experts defaults to 0 — reproduces v2 MoE
    exactly when unset)
[x] New fine-grained + shared-expert configs
      configs/medium_finegrained_moe.toml
      configs/large_finegrained_moe.toml
```

**`aarambh-studio-train`:**
```
[x] Router warm-start: initialise the finer-grained router from the
    already-trained v2 coarse router's weights where dimensions allow,
    rather than training routing from scratch
[x] Expert-count vs quality sweep script: trains small runs across a range
    of (num_experts, fine_grained_factor, top_k) combinations and logs
    eval-harness deltas, mirroring the MiniMax-M2.7-style observation that
    fine-grained routing can beat coarse routing at equal active-parameter
    budgets
```

### Tests

```rust
#[test]
fn moe_config_defaults_reproduce_v2_dense_moe_exactly() {
    // fine_grained_factor=1, num_shared_experts=0 -> byte-identical to
    // Phase 22's behaviour.
}

#[test]
fn shared_expert_output_is_added_for_every_token_unconditionally() {}

#[test]
fn shared_expert_excluded_from_load_balancing_auxiliary_loss() {}

#[test]
fn total_ffn_capacity_roughly_conserved_across_fine_grained_factor_values() {
    // Splitting into more, smaller experts should not silently change the
    // total parameter budget by more than the documented tolerance.
}

#[test]
fn router_warm_start_loads_compatible_dimensions_from_v2_checkpoint() {}
```

### Milestone
```
Fine-grained + shared-expert MoE implementation, matched coarse/fine configs,
function-preserving warm-start, and eval-harness sweep automation complete.
Hardware runs write their measured scorecards and comparisons without bundling
pretrained artifacts or claiming unexecuted results; the method and result
layout are documented in docs/phase31_moe_sweep.md.

git commit -m "feat: Phase 31 — fine-grained MoE routing with shared expert"
git tag v3.0.0-alpha.3
```

---

## Phase 32 — Multi-Token Prediction (MTP)

**Duration:** 7–10 days | **Hardware:** Kaggle (free quota)

### Goal
Add auxiliary prediction heads that predict multiple future tokens per
position during training (not just the next token), giving a denser
training signal per forward pass and a reusable draft mechanism for
inference-time speedups — and providing the auxiliary supervision that
Phase 33's on-policy distillation builds on.

### Tasks

**`aarambh-studio-core`:**
```
[x] Optional `MtpConfig` with explicit total-horizon semantics, defaults,
    serde compatibility, and range validation
```

**`aarambh-studio-nn`:**
```
[x] src/mtp.rs
      MtpHead — lightweight additional transformer block per future-token
      offset with the main LM projection shared, each conditioned on the trunk
      state plus the (real or previously-predicted) token embeddings for
      intervening positions
      MtpHead::forward() supports training and speculative proposal paths;
      heads are bypassed during ordinary next-token inference
```

**`aarambh-studio-train`:**
```
[x] src/mtp_loss.rs
      Auxiliary loss: weighted sum of the main next-token loss and each
      MTP head's future-token loss, weight configurable and typically
      small relative to the main loss (auxiliary signal, not the primary
      objective)
      Loss logging reports main-loss and each MTP-head loss separately in
      the training scorecard
```

**`aarambh-studio-inference`:**
```
[x] MTP heads optionally reused as the draft model for Phase 25's
    speculative decoding (v2 §29) — since they already predict several
    tokens ahead from the same trunk, they are a natural free draft
    source, avoiding the need for a separate small draft checkpoint in
    configs where MTP is enabled
```

**Configuration, checkpoints, and tooling:**
```
[x] `MtpConfig` is optional and backward-compatible; its horizon includes
    the main t+1 head, so MTP-2 creates exactly one auxiliary head
[x] SafeTensors/GGUF preserve all heads; retrofit fresh-initializes a complete
    absent set and rejects partial MTP checkpoints
[x] Tiny CPU smoke plus Medium/Large Phase 31 continuation recipes
[x] Matched training comparison and exact greedy throughput benchmark scripts
[x] `--speculative` selects internal MTP with no draft paths and retains the
    existing external draft model path when those paths are supplied
```

### Tests

```rust
#[test] // complete
fn mtp_head_output_shape_matches_num_future_tokens_config() {}

#[test] // complete
fn mtp_disabled_config_trains_identically_to_pre_v3_next_token_only_loss() {}

#[test] // complete
fn mtp_auxiliary_loss_does_not_dominate_main_loss_at_default_weight() {}

#[test] // complete
fn mtp_heads_usable_as_speculative_draft_source_without_separate_checkpoint() {}
```

### Milestone
```
MTP model, loss, checkpoint, retrofit, and one-checkpoint exact speculative
decoding paths are complete. CPU smoke tests prove execution and exact greedy
equivalence. `scripts/phase32_compare_training.sh` and
`scripts/phase32_benchmark_mtp.sh` produce quality and throughput evidence on
real trained checkpoints; repository documentation does not claim unexecuted
hardware gains.

git commit -m "feat: Phase 32 — multi-token prediction heads"
git tag v3.0.0-alpha.4
```

---

## Phase 33 — On-Policy Distillation

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

### Goal
A distillation pipeline where the student model is trained on its own
on-policy rollouts scored/corrected by a larger teacher (rather than
static teacher-generated data), reducing the train/inference distribution
mismatch that plain offline distillation suffers from.

### Tasks

**New crate `aarambh-studio-distill`:**
```
[x] src/rollout.rs
      Student generates completions for a batch of prompts using its own
      current weights (on-policy) — reuses aarambh-studio-inference's decode
      path directly, no separate generation code

[x] src/teacher_score.rs
      TeacherScorer trait — abstracts over "teacher" being either a larger
      local aarambh-studio checkpoint or a scored-reference dataset; scores or
      corrects the student's own rollouts rather than only supplying
      independently-generated teacher text
      KL-style or reward-style scoring paths, both supported behind the
      trait so the same distillation loop works with either teacher kind

[x] src/distill_loss.rs
      On-policy distillation loss: student is trained to match
      teacher-assigned quality on its *own* generated sequences, blending
      with the MTP auxiliary loss (Phase 32) where enabled
```

**`aarambh-studio-distill` trainer using `aarambh-studio-train` primitives:**
```
[x] Distillation training loop: alternates rollout generation (inference
    mode, no gradient) with gradient updates on the scored rollouts
    (training mode) — structurally similar to v1's Online GRPO loop
    (`SELF_LEARNING.md` §5) but using teacher scoring instead of
    verifier-based reward
```

### Tests

```rust
#[test] // complete
fn rollout_generation_reuses_inference_decode_path_unmodified() {}

#[test] // complete
fn teacher_scorer_trait_accepts_both_local_checkpoint_and_dataset_backends() {}

#[test] // complete
fn distill_loss_gradient_flows_only_through_student_not_teacher() {}

#[test] // complete for the KL proxy; full checkpoints use the comparison harness
fn on_policy_distillation_reduces_train_inference_distribution_gap_metric() {
    // Proves optimization reduces teacher/student KL on replay positions;
    // fresh-rollout checkpoint comparison remains a hardware benchmark.
}
```

### Milestone
```
The full local-teacher, scored-dataset, offline-control, evaluation, checkpoint,
and exact-resume paths pass the two-step CPU release smoke. Medium/Large
quality is measured by `scripts/phase33_compare_distillation.sh` at a matched
optimizer-update budget; `docs/phase33_distillation_results.md` defines the
acceptance rule and does not claim an unexecuted Kaggle result.

git commit -m "feat: Phase 33 — on-policy distillation pipeline"
git tag v3.0.0-alpha.5
```

---

## Phase 34 — Native QAT (Quantization-Aware Training)

**Duration:** 7–10 days | **Hardware:** i3 (small scales) + Kaggle (larger)

### Goal
Fold quantization into training itself — rather than v1's post-hoc
INT4/GGUF conversion (`ARCHITECTURE.md` §16) applied only after training
finishes, QAT simulates quantization noise during training so the model
learns weights that are robust to it, matching the 2026 pattern of
shipping official quantized variants as part of the release itself, not
an afterthought.

### Tasks

**`aarambh-studio-quant`:**
```
[x] Device-native FakeQuantize for INT4/INT8 with identity STE; no host
    tensor conversion in the training forward/backward path
[x] ExportAligned mode exactly matches Q4_K_M blocks (256 values, f16
    scale/min, padded tails) and global Q8 absmax; DSA indexers remain Q8
[x] PerTensor and PerOutputChannel alternatives plus explicit QatTarget set
[x] QatLinear preserves Candle's contiguous matmul fast paths and caches one
    effective weight per optimizer generation
[x] Calibration orchestration moved to the CLI so quant remains below model
    assembly and can be depended on by aarambh-studio-nn/model
```

**`aarambh-studio-model`, `aarambh-studio-nn`, and `aarambh-studio-train`:**
```
[x] QAT wraps attention, FFN/expert, MoE-router, DeltaNet, DSA-indexer, MTP,
    and optional LM-head projections; embeddings/norms/convolutions/scalars
    remain full precision
[x] QAT activates only through AarambhModel::new_for_training; ordinary
    model loading and inference never add fake quantization implicitly
[x] Optimizer steps and model-only loads advance the cache generation
[x] Exact SafeTensors initialization rejects missing, unexpected, or
    shape-mismatched tensors
[x] TrainState persists QatConfig and exact resume rejects policy changes
[x] QAT logs bit width, granularity, coverage, generation, and refresh count
[x] Post-QAT conversion reuses the existing GGUF exporter unchanged
```

**Evaluation and tooling:**
```
[x] eval --qat-compare emits baseline FP, baseline quantized, QAT FP, and
    QAT quantized scorecards plus direction-normalized drop/recovery
[x] configs/qat_smoke.toml and configs/qat_tiny.toml
[x] scripts/phase34_smoke.sh and scripts/phase34_compare_qat.sh
[x] Criterion coverage for fake quantization, cached forwards, and refreshes
```

### Tests

```rust
#[test] // complete against Q4_K_M/Q8, including padded tails
fn fake_quantize_forward_matches_post_hoc_quantization_numerically() {}

#[test] // complete; identity STE is asserted exactly
fn fake_quantize_straight_through_backward_produces_finite_gradients() {}

#[test] // implemented as a four-scorecard eval acceptance gate
fn qat_trained_checkpoint_loses_less_eval_score_than_post_hoc_quantized_baseline() {
    // Requires real baseline/QAT checkpoints; source tests validate report
    // direction math without claiming an unexecuted training result.
}

#[test] // complete: normal construction ignores QAT and old configs default off
fn qat_config_default_off_reproduces_v1_full_precision_training_exactly() {}
```

### Milestone
```
Native INT4/INT8 QAT is complete in the shared training path with exact export
parity, STE gradients, generation caching, strict continuation, CPU smoke,
benchmarking, and a reproducible four-way quality gate. Small/Medium quality
acceptance remains checkpoint evidence produced by `eval --qat-compare`; no
pretrained checkpoint or unexecuted gain is claimed by this source release.

git commit -m "feat: Phase 34 — native quantization-aware training"
git tag v3.0.0-alpha.6
```

---

## Phase 35 — Native Video Understanding

**Duration:** 14–18 days | **Hardware:** Kaggle (free quota)

### Goal
Extend v2's frozen-ViT vision pipeline (`ARCHITECTURE_V2.md` §24–25) to
video: sample and encode frames, fuse them with temporal position
information, and train on free public video-QA data — built in as a
first-class modality path from this phase forward rather than adapted
onto the text model as an afterthought, matching how MiniMax M3 and
Qwen3.5 integrate video/vision early rather than bolting it on.

### Tasks

**`aarambh-studio-vision` (extends v2's crate):**
```
[x] src/video.rs
      Native H.264 MP4 decode through bundled OpenH264, deterministic
      uniform or scene-aware fixed-count frame sampling, and a bounded
      cache for detached frozen-encoder features

[x] src/temporal.rs + src/video_fusion.rs
      Learned or sinusoidal temporal offsets, exact-zero frame zero, and
      interleave_video_tokens() with explicit frame-separator validation

[x] src/video_data.rs
      Normalized JSONL VideoQaExample loading plus direct official NExT-QA
      CSV parsing and multiple-choice target normalization

[x] src/preprocess.rs
      CPU frame preprocessing followed by one contiguous batch transfer;
      frozen CLIP forwards are chunked by encoder_frame_batch_size
```

**`aarambh-studio-finetune`:**
```
[x] Video instruction tuning reuses v2 Phase 20's DoRA-adapted VLM
    training path (`vlm_dora.rs`), extended to accept a sequence of frame
    embeddings per example instead of a single image's embeddings; learned
    temporal parameters share accumulation, clipping, and artifact-save cadence
```

**`aarambh-studio-tokenizer`:**
```
[x] New reserved special tokens: <video>, <video_end>, <frame_sep> — IDs
    9, 10, and 11, with deterministic legacy tokenizer and SafeTensors
    vocabulary migration while preserving all existing token text
```

### Data Setup

```bash
# Deterministic four-clip local fixture. FFmpeg is used only to create the
# fixture; the aarambh-studio runtime decoder never invokes it.
python3 scripts/phase35_make_video_smoke_fixture.py
scripts/phase35_smoke.sh

# Real training/eval accepts normalized JSONL or official NExT-QA CSV
# directly; point [vision.video].video_root at the extracted H.264 MP4 clips.
```

### Tests

```rust
#[test] // exact-count uniform/scene samplers and native decode smoke
fn frame_sampler_returns_configured_frame_count_for_variable_length_video() {}

#[test] // complete for learned and sinusoidal positions
fn temporal_position_embedding_distinguishes_frame_order() {
    // Shuffling frame order changes the fused embedding sequence, proving
    // temporal information is actually encoded, not just concatenated.
}

#[test] // complete; one-frame temporal path is identity
fn interleave_video_tokens_extends_image_interleaving_without_regressing_it() {
    // Single-frame ("image") case must still behave exactly as v2's
    // interleave_image_tokens() did.
}

#[test] // one shared trainer handles Image and Video examples
fn video_qa_instruction_tuning_reuses_vlm_dora_path_without_duplication() {}
```

### Milestone
```
Native video ingestion, fixed-count sampling, temporal fusion, shared VLM DoRA
training, CLI inference, and `eval --tasks video-qa|nextqa` are complete. The
checked-in smoke proves execution and metric plumbing; useful held-out NExT-QA
accuracy remains evidence produced by training a real checkpoint and is not
claimed by this source-only release.

git commit -m "feat: Phase 35 — native video understanding"
git tag v3.0.0-alpha.7
```

---

## Phase 36 — Native Document Understanding

**Duration:** 10–14 days | **Hardware:** Kaggle (free quota)

### Goal
Extend the same vision pipeline to documents (PDFs, scanned pages,
multi-column layouts, tables) with layout-aware encoding, sharing the
frozen ViT encoder and fusion path from Phases 19/20/35 rather than
standing up a separate document-specific model.

### Tasks

**`aarambh-studio-vision` (extends the crate again):**
```
[x] src/document_sample.rs
      PageRasterizer — renders PDFs through pinned pure-Rust Hayro 0.4 and
      accepts ordered page images for the
      existing frozen ViT encoder path (documents are treated as a
      sequence of page-images, reusing v2's per-image encoder rather than
      a new architecture); defaults to 150 DPI and at most 16 pages, with
      explicit pixel limits and a bounded detached-feature cache
      Reuses `image` crate preprocessing (v2 §32 dependency policy);
      document-to-image rendering uses a permitted crate, documented below

[x] src/layout_projector.rs
      LayoutAwareProjector — extends v2's plain Projector MLP (§24) with
      2D positional information per patch (row/column position on the
      page), so the model can distinguish "this text is in a table cell"
      from "this text is a paragraph," which a plain sequential ViT patch
      stream loses
      Table/multi-column structure is not separately parsed/OCR'd —
      the model learns layout from position-augmented patches plus
      instruction-tuning data, consistent with the "vision-language
      reasoning" framing of v2's VLM approach rather than a hybrid
      OCR+LLM pipeline

[x] src/instruct_data.rs (extends again)
      DocQaExample — JSONL schema for document-question-answer pairs,
      covering both born-digital PDFs and scanned/rasterized pages
```

**`aarambh-studio-finetune`:**
```
[x] Document instruction tuning reuses the same DoRA-adapted VLM path as
    Phase 35's video tuning and v2 Phase 20's image tuning — one training
    code path, three data types (image, video, document)
```

**`aarambh-studio-tokenizer`:**
```
[x] New reserved special tokens: <document>, <document_end>, <page_sep> —
    IDs 12, 13, and 14, with deterministic video-tokenizer and SafeTensors
    migration
```

### Data Setup

```bash
# Deterministic four-PDF local fixture and complete smoke workflow.
python3 scripts/phase36_make_document_smoke_fixture.py
scripts/phase36_smoke.sh

# Normalize annotations from a user-downloaded DocVQA/MP-DocVQA release.
# Dataset login, terms, and original files remain the user's responsibility.
python3 scripts/phase36_prepare_docvqa.py \
  --annotations /path/to/annotations.json \
  --documents-dir /path/to/documents \
  --output data/document_qa/train.jsonl
```

### Tests

```rust
#[test]
fn page_rasterizer_produces_consistent_image_size_across_page_orientations() {}

#[test]
fn layout_aware_projector_encodes_2d_position_distinctly_from_1d_sequence_position() {}

#[test]
fn document_qa_handles_multi_page_documents_via_page_sep_tokens() {}

#[test]
fn table_cell_text_and_paragraph_text_produce_distinguishable_embeddings() {
    // Proxy test: same text string rendered in a table cell vs a
    // paragraph should not collapse to identical fused embeddings.
}
```

### Milestone
```
Native PDF/page ingestion, layout-aware projection, shared DoRA/QDoRA tuning,
CLI inference, and `eval --tasks document-qa|docvqa` with ANLS are complete.
The checked-in multi-page smoke fixture proves execution and metric plumbing;
useful held-out DocVQA accuracy remains evidence produced by training a real
checkpoint and is not claimed by this source-only release.

git commit -m "feat: Phase 36 — native document understanding"
git tag v3.0.0-alpha.8
```

---

## Phase 37 — Long-Horizon Tool-Use Chains

**Duration:** 10–14 days | **Hardware:** i3 (orchestration) + Kaggle (training)

### Goal
Upgrade v2 Phase 26's single-call, emit-only tool use
(`ROADMAP_V2.md` Phase 26, `ARCHITECTURE_V2.md` §30) into multi-step
chains: the model can call a tool, receive a result, decide whether
another call is needed, and continue — sustained across many steps rather
than one typed request per turn.

### Tasks

**New crate `aarambh-studio-agent`:**
```
[x] src/chain.rs
      ToolChain — orchestrates repeated calls into aarambh-studio-inference's
      existing single-call tool-call decoding path (v2 §30), feeding each
      tool's result back into context as the next turn's input
      MaxSteps / stopping conditions — explicit step budget, explicit
      "no further tool needed" detection reusing v2's existing fallback
      path where the model emits a normal (non-tool-call) response

[x] src/result_ingestion.rs
      ToolResult — typed wrapper for a tool's returned value, formatted
      back into the model's context in a consistent schema; supports
      text, and (building on Phases 35–36) image/video/document results
      so a chain step can hand back "here is a screenshot" or "here is a
      retrieved PDF page" and have the next step reason over it natively

[x] src/state.rs
      ChainState — the accumulating context across steps (prior calls,
      results, and the running conversation), with an explicit eviction/
      summarisation policy once the chain approaches context-length limits
      (reuses Phase 29/30's long-context attention stack, but even hybrid
      attention has a practical ceiling worth planning for)
```

**`aarambh-studio-finetune`:**
```
[x] Multi-step tool-use SFT: extends v2 Phase 26's tool_sft.rs loss-
    masking scheme (mask everything except the tool-call spans and final
    response) across multi-turn transcripts, so the model is trained on
    realistic multi-call sequences, not just single-call examples
```

**CLI:**
```
[x] aarambh-studio agent --config <cfg> --tools tools.json --prompt "..." --max-steps 8
```

### Tests

- [x] Normal final responses stop the chain.
- [x] Repeated calls cannot exceed the configured `1..=64` call budget.
- [x] Text/error/image/video/document result envelopes are validated.
- [x] Context pressure evicts the oldest unprotected exchange.
- [x] Multi-step SFT supervises every tool-call span and final response while
      masking prompts, results, and optional thinking context.
- [x] Scripted replay rejects call-id, call-name, or argument mismatches.

### Milestone
```
The source implementation now includes `eval --tasks tool-chain` with ordered
call/schema/argument/final-answer metrics, a checked-in three-call response
path, and a BFCL v1.3 explicit-response-path normalizer. The max-step ceiling
is enforced in code. Useful held-out chain success is a checkpoint-quality
result that must be produced by multi-step training and evaluation; it is not
claimed by this source-only alpha.

Still an emit/orchestrate boundary consistent with v2 §30's framing: tool
*execution* itself remains the caller's responsibility (the chain
orchestrates typed requests and ingests typed results, it does not sandbox
or authorize arbitrary tool execution).

git commit -m "feat: Phase 37 — long-horizon tool-use chains"
git tag v3.0.0-alpha.9
```

---

## Phase 38 — Forgetting Diagnostics Tied to Manas

**Duration:** 7–10 days | **Hardware:** i3 (small scales) + Kaggle (larger)

### Goal
A diagnostic toolkit that measures catastrophic forgetting through
controlled, fixed capability-probe regressions rather than unrelated
one-off eval runs, producing persistent per-capability curves that both
aarambh-studio's own self-learning loop (`SELF_LEARNING.md` §8,
`SELF_LEARNING_V2.md` §17) and Manas v3's anti-forgetting design
(`SELF_LEARNING_V3.md` in the Manas repo) can consume as a shared signal.

### Tasks

**`aarambh-studio-eval` (extends v2's crate):**
```
[x] src/forgetting.rs
      CapabilityProbe — a small, fixed held-out set per capability
      (math, code, reasoning, factual, vision, video, document, tool-use),
      reusing the eval harness's existing task subsets (v2 §17) as probes
      rather than inventing new benchmark data
      ForgettingCurve — tracks a capability probe's score across training
      checkpoints/steps over time, not just before/after a single phase
      forgetting_delta() — signed score change per capability between any
      two checkpoints, with a documented significance threshold below
      which noise (not real forgetting) is assumed
      ProbeManifest — validated, fingerprinted ownership of fixed task
      subsets; unavailable modality/permission probes are explicit skips
      unless strict mode is requested
      ForgettingStore — atomic, restart-safe, idempotent multi-point storage
      keyed by suite and tokenizer fingerprints

[x] src/report.rs (extends v2's Scorecard)
      Scorecard gains a per-capability forgetting section alongside the
      existing absolute scores, exported to the same markdown/JSON formats
[x] MoE routing signatures captured per probe example and compared as a
    separate routing-drift diagnostic; dense checkpoints pay no routing
    introspection cost
```

**`aarambh-studio-selflearn`:**
```
[x] Forgetting diagnostics wired into the existing self-learning loop
    (`SELF_LEARNING.md` §5, §8): after each online-GRPO update batch, run
    the lightweight capability probes and log forgetting_delta() per
    capability. Deferred CPU gradients are measured only when they are
    flushed; replay is measured after its committed optimizer update.
[x] Shared export format: forgetting curves exported in a schema
    documented to be directly importable by Manas's associative-memory
    anti-forgetting tracking (cross-project consistency between
    aarambh-studio and Manas's own forgetting-diagnostics work)
[x] No runtime dependency or filesystem discovery of `../manas`; JSONL is
    an explicit optional bridge controlled by the caller
```

**Training and CLI integration:**
```
[x] Read-only training observer runs a baseline, configurable optimizer-step
    probes, and a final probe without checkpoint serialization or parameter
    mutation; distributed ranks synchronize around rank-0 diagnostics
[x] `aarambh-studio eval` records named checkpoint/session points and can export
    JSON/Markdown scorecards plus the seven-field bridge JSONL
[x] `aarambh-studio selflearn forgetting-report` summarizes persistent curves
[x] Checked-in probe manifest, JSON Schema, preparation/smoke scripts,
    smoke training config, and Phase 38 operating guide
```

### Tests

```rust
#[test]
fn capability_probe_reuses_eval_harness_task_subsets_without_duplicating_data() {}

#[test]
fn forgetting_curve_tracks_score_across_multiple_checkpoints_not_just_two() {}

#[test]
fn forgetting_delta_below_significance_threshold_is_not_flagged() {}

#[test]
fn forgetting_diagnostics_do_not_alter_training_gradients_diagnostic_only() {
    // This phase measures forgetting, it does not by itself change how
    // training proceeds or introduce a new anti-forgetting algorithm.
}

#[test]
fn forgetting_export_schema_matches_documented_manas_import_format() {}
```

### Milestone
```
Opt-in forgetting diagnostics run as part of both a standard training loop
and the self-learning loop, producing persistent per-capability curves across
named checkpoints/sessions. Diagnostics are strictly measurement-only: the
implemented safeguards remain frozen base weights, LoRA/DoRA updates, KL
regularization, small learning rates, and diverse replay. The exact seven-field
JSONL bridge is schema-validated and can be imported by Manas without creating
a source, runtime, or filesystem dependency between the projects.

git commit -m "feat: Phase 38 — forgetting diagnostics tied to Manas"
git tag v3.0.0-alpha.10
```

---

## Phase 39 — Max Thinking Mode

**Duration:** 5–7 days | **Hardware:** i3 (inference) + Kaggle (retraining the thinking-allocation policy)

### Goal
A fifth thinking mode, **Max**, added above v1's None (0) → Low (≤256) →
Medium (≤1,024) → High (≤4,096) token-budget lineup
(`ARCHITECTURE.md` §7.2), for tasks that genuinely need more room to
reason than High allows — long multi-stage proofs, dense multi-file code
changes, or (building directly on Phase 37) planning across an entire
long-horizon tool-use chain before the first tool call is even made.

### Tasks

**`aarambh-studio-inference`:**
```
[x] src/thinking.rs (extends v1's existing module)
      ThinkingMode gains a fifth variant: None / Low(256) / Medium(1024)
      / High(4096) / Max(16384) — budget chosen as the next step in the
      existing ~4x progression between modes, not an arbitrary number
      ThinkingController's existing force-close-at-budget logic
      (`take_forced_token()`, `on_token()`, `ARCHITECTURE.md` §7.4)
      requires no structural change — Max is just another budget value
      flowing through the same mechanism that already handles the other
      four modes
      (Phase 39 also centralised FromStr/Display on ThinkingMode and added
      default_sampler() for the per-mode sampling table.)
```

**`aarambh-studio-train`:**
```
[x] Sampling defaults for Max mode, extending v1's existing table
    (`ARCHITECTURE.md` §8.2):
      Thinking mode Max: temperature=0.85, top_p=0.97 (most exploratory
      of the five — Max-mode tasks are exactly the ones where premature
      convergence on a wrong early step is most costly)
[x] Stage 2 GRPO re-run (`ARCHITECTURE.md` §7.5) extended to include
    Max-budget rollouts in the G=8-completions-per-problem sampling, on a
    held-out set of problems specifically hard enough that High's 4,096-
    token budget was previously insufficient to reach a correct answer —
    without such problems in the training mix, the model has no signal
    for when spending up to 16,384 tokens is actually worth it
    (Optional Kaggle helper: scripts/phase39_kaggle_grpo.sh; requires a
    trained checkpoint and is not run by CI.)
    GrpoThinkingMode::Max (budget 16,384) is accepted by GrpoConfig.thinking,
    the GRPO CLI (--thinking max), and the self-learning GRPO loop; the
    LocalThinkingState in grpo.rs forces start/close markers and tracks the
    Max budget via the same mechanism as lower modes.
[x] Reuses the existing format verifier and reward shaping unchanged
    (correct + concise thinking → high reward; wrong answer → negative
    reward; excessive empty thinking → penalised) — Max mode does not
    get a separate reward function, only a larger budget ceiling within
    the same incentive structure
```

**`aarambh-studio-eval`:**
```
[x] New eval-harness task: a held-out "hard problems" subset specifically
    selected because they are unsolved (or solved at low accuracy) under
    High-mode budget, scoring Max-mode accuracy against that same set —
    the direct test of whether Max mode earns its larger budget rather
    than just spending more tokens for the same outcome
    (data/eval/hard_problems/data.jsonl + crates/aarambh-studio-eval/src/tasks/hard_problems.rs;
    reports accuracy, thinking_tokens, completion_tokens, total_tokens.)
```

**CLI:**
```
[x] aarambh-studio infer --config <cfg> --thinking max --prompt "..."
[x] aarambh-studio agent --config <cfg> --thinking max --tools tools.json ...
      # pairs naturally with Phase 37's tool chains: Max-budget planning
      # before the first tool call, on the hardest multi-step tasks
    (serve, finetune grpo, distill train, selflearn start, and eval also
    accept --thinking max via the same centralised parser.)
```

### Tests

```rust
#[test]
fn thinking_mode_max_budget_is_16384_tokens() {}

#[test]
fn thinking_controller_force_closes_max_mode_at_budget_exactly_like_other_modes() {
    // No special-cased logic path for Max — same on_token()/
    // take_forced_token() mechanism as None/Low/Medium/High.
}

#[test]
fn max_mode_sampling_defaults_are_more_exploratory_than_high_mode() {}

#[test]
fn max_mode_accuracy_on_high_mode_unsolved_holdout_exceeds_high_mode_baseline() {
    // The actual point of this phase: Max earns its budget on problems
    // where High previously fell short.
}

#[test]
fn existing_none_low_medium_high_modes_are_byte_for_byte_unchanged() {}
```

### Milestone
```
DONE (3.0.0). Max mode shipped as a fifth ThinkingMode variant with
zero structural changes to ThinkingController — same forced-token mechanism,
same budget-tracking, same collapse-on-force-close behaviour every existing
mode already has. Parsing/display centralised on ThinkingMode; per-mode
sampling defaults added; hard-problems eval task added; infer/agent/serve/
finetune grpo/distill train/selflearn start/eval all accept --thinking max.
Accuracy improvement over High mode on a held-out High-insufficient set is
documented (schema, no invented numbers) in docs/phase39_max_thinking_results.md;
the optional Kaggle helper scripts produce the measurement when a trained
checkpoint is supplied. `aarambh-studio infer --thinking max` and
`aarambh-studio agent --thinking max` both work end to end.

git commit -m "feat: Phase 39 — Max thinking mode (16,384-token budget)"
git tag v3.0.0-alpha.11
```

---

## Phase 40 — v3.0.0 Source Release

**Duration:** 5–7 days | **Hardware:** all

### Goal
Ship the v3.0.0 source release: documentation, release artifacts, and CI
extensions — **source code only, zero model artifacts.** No pretrained
checkpoints, adapters, or GGUF files are attached to this release. crates.io
publishing is deferred to v4.0.0.

### Tasks

```
[x] CHANGELOG.md: v3.0.0 entry summarising Phases 29–39
[x] README.md, ARCHITECTURE.md, ROADMAP.md, SELF_LEARNING.md: merge in the
    v3 additions (or keep _V3 docs as addenda, same option v2 documented)
[x] RELEASE.md: v3.0.0 checklist, explicitly restating "no pretrained
    checkpoints, no model artifacts" for this release line too
[x] .github/release-notes/v3.0.0.md
[x] CI: extend existing workflow to cover the 2 new crates + agent/eval
    smoke runs; update release workflow for v3.0.0 tag
[x] All public APIs documented (cargo doc passes with -D missing_docs)
[x] All workspace packages set to version 3.0.0, publish = false
```

### Milestone
```
cargo install --path aarambh-studio
aarambh-studio --version  → aarambh-studio 3.0.0
git tag v3.0.0
git push origin v3.0.0

All 19 workspace packages share version 3.0.0 and remain
publish = false. No model weights attached to the GitHub Release.

git commit -m "chore: v3.0.0 — source release"
```

---

## Complete Phase Summary

| # | Phase | Key Deliverable | Hardware | Duration |
|---|---|---|---|---|
| 29 | Gated DeltaNet | Hybrid linear attention, retrofit via continued pretraining | Kaggle | 10–14 days ✅ |
| 30 | DSA | Sparse attention for remaining full-attention layers | Kaggle | 10–14 days ✅ |
| 31 | Fine-Grained MoE | DeepSeek-style routing + shared expert, upgrades v2 dense MoE | Kaggle | 10–14 days ✅ |
| 32 | MTP | Multi-token prediction heads, doubles as speculative-decode draft | Kaggle | 7–10 days ✅ |
| 33 | On-Policy Distillation | New `aarambh-studio-distill`, teacher-scored student rollouts | Kaggle | 10–14 days ✅ |
| 34 | Native QAT | Fake-quantize training, folds INT4/INT8 into the training loop | i3 + Kaggle | 7–10 days ✅ |
| 35 | Video Understanding | Frame sampling + temporal fusion, extends `aarambh-studio-vision` | Kaggle | 14–18 days ✅ |
| 36 | Document Understanding | Layout-aware projector, shares vision encoder with video | Kaggle | 10–14 days ✅ |
| 37 | Long-Horizon Tool Chains | New `aarambh-studio-agent`, multi-step tool calls with result ingestion | i3 + Kaggle | 10–14 days ✅ |
| 38 | Forgetting Diagnostics | Per-capability forgetting curves, shared export format for Manas | i3 + Kaggle | 7–10 days ✅ |
| 39 | Max Thinking Mode | 5th reasoning depth, 16,384-token budget, extends `ThinkingController` | i3 + Kaggle | 5–7 days ✅ |
| 40 | v3.0.0 Source Release | CHANGELOG, RELEASE.md, CI, docs, release notes; crates.io deferred to v4 | all | 5–7 days ✅ |

**Total realistic estimate: 105–145 days (~3.5–4.8 months)**

---

## Dependency Policy Additions (v3.0)

| Dependency | Allowed crates | Reason |
|---|---|---|
| video container decode crate (permissive-licensed, e.g. an `ffmpeg`-free pure-Rust decoder) | `aarambh-studio-vision` | Frame extraction only, no network calls |
| PDF/document rasterisation crate | `aarambh-studio-vision` | Page-to-image rendering only, no network calls |
| (no new dependency for distillation or agent chains — both reuse existing `candle-core` and inference paths) | `aarambh-studio-distill`, `aarambh-studio-agent` | — |

**Still forbidden everywhere, unchanged from v1/v2:** PyTorch bindings
(`tch-rs`), ONNX Runtime (`ort`), Python FFI, `llama.cpp` as a backend. All
computation goes through `candle`. Video and document rasterisation
crates must be pure-Rust or bind to a permissively-licensed system library
already compatible with the "no PyTorch/ONNX/Python FFI" rule — no
dependency on Python-based video/PDF ML tooling.

**Version policy:** unchanged from v1/v2 — pin major versions, test the
whole workspace on any `candle-core` upgrade.

---

## What's Explicitly Out of Scope for v3.0

- Releasing any pretrained checkpoint, adapter, or GGUF file
- Sparse/grouped MoE dispatch (still dense-masked-matmul, per v2 §35 —
  carried forward again; genuine sparse dispatch remains a documented
  future optimisation)
- Multi-node distributed training (still 2-GPU single-node only, per v2's
  Phase 23 scope)
- Public/hosted deployment of the inference server (still local-only, per
  v2's Phase 27 scope)
- Audio modality (video understanding in Phase 35 is visual frames only —
  no audio track processing; a natural v4 candidate)
- Tool *execution*/sandboxing (Phase 37 orchestrates and ingests results,
  it does not execute or authorize tools itself — same emit-only
  boundary v2 §30 established, extended to multi-step, not removed)
- Synthetic API-generated training data (all v3.0 datasets remain free and
  public, same policy as every prior phase)

These are natural v4 candidates once budget and a released model exist.
