# ROADMAP_V2.md — aarambh-ai v2.0

> Step-by-step build plan for v2.0. Every phase ends with working, testable code.
> Builds on the completed v1.0.0 base (Phases 0–15, all ✅). No pretrained
> checkpoints are released as part of v2.0 — this is a source/engineering
> release, same policy as v1.0.0.

---

## How to Read This Roadmap

Each phase has:
- **Goal** — exactly what you will have when this phase is done
- **Tasks** — the checklist to follow, in order, grouped by crate
- **Tests** — what you write to prove it works
- **Milestone** — how you know you are done, with the git tag to cut

Work top to bottom. Do not skip phases — each phase depends on the ones
before it. Phases 16–18 are pure engineering with no external data cost.
Phases 19–21 (vision) need only free public datasets. All Kaggle phases use
the free weekly GPU quota; none require paid API access.

---

## Phase Map (Quick Reference)

```
Phase 16 →  Long context (RoPE scaling)        (7–10 days)   [Kaggle, free]
Phase 17 →  Evaluation harness                 (7–10 days)   [i3 + Kaggle]
Phase 18 →  DoRA fine-tuning                    (7–10 days)   [i3 + Kaggle]
Phase 19 →  Vision encoder + projector          (10–14 days)  [Kaggle] ✅
Phase 20 →  Vision-language training            (10–14 days)  [Kaggle]
Phase 21 →  Vision-aware self-learning          (7–10 days)   [Kaggle only]
Phase 22 →  Mixture of Experts                  (10–14 days)  [Kaggle]
Phase 23 →  Multi-GPU training                  (7–10 days)   [Kaggle 2×T4]
Phase 24 →  DPO preference tuning               (7–10 days)   [Kaggle]
Phase 25 →  Speculative decoding                (5–7 days)    [Kaggle]
Phase 26 →  Tool use / function calling         (7–10 days)   [i3 + Kaggle]
Phase 27 →  Inference server                    (10–14 days)  [i3]
Phase 28 →  crates.io publish (v2.0.0)          (5–7 days)    [all]
```

**Total realistic estimate: 99–140 days (~3.3–4.7 months)**

---

## Why This Order

1. **16–18 first** — long context, eval harness, and DoRA are all pure
   crate-level engineering. None of them need a new dataset type, none need
   more than one Kaggle GPU, and the eval harness (17) gives you a real
   measuring stick for everything that follows. Do these before anything
   riskier.
2. **19–21 (vision)** comes next because it's the largest architectural
   addition and benefits most from having eval (17) and DoRA (18) already in
   place — DoRA is reused directly for VLM instruction tuning in Phase 20,
   and eval tells you whether the projector actually learned anything.
3. **22–23 (MoE, multi-GPU)** come after vision because they are training-
   infrastructure changes best validated on the now-larger, now-measurable
   model, and multi-GPU work directly benefits the heavier MoE and VLM
   training jobs that follow it in practice.
4. **24–26 (DPO, speculative decoding, tool use)** are refinements that only
   make sense once the base model (text + vision, evaluable) is stable.
5. **27 (server)** is deliberately near the end — it should serve a model
   with the full v2 feature set, not an early prototype. It serves your own
   local checkpoints only; no weights are published by this phase.
6. **28 (crates.io)** is last, exactly like v1.0.0's Phase 15 discipline:
   ship the *code* as source once it is proven, never ship unproven code
   or unreleased weights.

---

## Workspace `Cargo.toml` Additions

```toml
[workspace]
members = [
    # ...existing v1.0.0 members unchanged...
    "crates/aarambh-ai-core",
    "crates/aarambh-ai-tokenizer",
    "crates/aarambh-ai-data",
    "crates/aarambh-ai-nn",
    "crates/aarambh-ai-kernel",
    "crates/aarambh-ai-model",
    "crates/aarambh-ai-weights",
    "crates/aarambh-ai-quant",
    "crates/aarambh-ai-train",
    "crates/aarambh-ai-finetune",
    "crates/aarambh-ai-inference",
    "crates/aarambh-ai-safety",
    "crates/aarambh-ai-selflearn",

    # new in v2.0
    "crates/aarambh-ai-eval",       # Phase 17
    "crates/aarambh-ai-vision",     # Phase 19
    "crates/aarambh-ai-serve",      # Phase 27

    "aarambh-ai",
]
```

Three new crates (`aarambh-ai-eval`, `aarambh-ai-vision`, and
`aarambh-ai-serve`). Everything else extends existing crates. No new
external dependencies beyond what's listed in each phase's Dependency
Policy note.

---

## Phase 16 — Long Context (RoPE Scaling)

**Duration:** 7–10 days | **Hardware:** Kaggle (free quota)

### Goal
Extend usable context from the current 4,096-token ceiling (Large scale) to
16,384+ tokens without retraining from scratch, using YaRN-style RoPE
frequency interpolation. Short-context quality does not regress.

### Tasks

**`aarambh-ai-nn`:**
```
[x] src/rope_scaling.rs
      RopeScalingConfig lives in aarambh-ai-core because ModelConfig owns it;
      aarambh-ai-nn implements method: Yarn | Ntk | Linear, factor,
      original_max_seq_len, beta_fast, beta_slow, attn_factor
      yarn_frequencies() — per-dimension interpolation between original and
        scaled inverse frequencies, ramp function over beta_fast/beta_slow
      ntk_aware_theta() — simple alternative: rescale rope_theta by factor^(d/(d-2))
      RopeCache::from_config() — drop-in scaled cache builder for existing apply()

[x] src/attention.rs
      Causal attention dispatch avoids preallocating a max_seq_len x max_seq_len mask
      KV cache preallocation now takes scaled max_seq_len as a parameter
```

**`aarambh-ai-model`:**
```
[x] Model config gains `rope_scaling: Option<RopeScalingConfig>`
[x] Backward compatible: rope_scaling = None reproduces exact v1.0.0 output
[x] New long-context variants of Medium/Large configs with rope_scaling set
      configs/medium_16k.toml
      configs/large_16k.toml
```

**`aarambh-ai-train`:**
```
[x] Short continued-pretraining recipe: fine-tune an existing checkpoint on
    long-document data at the new context length (not from-scratch training)
[x] Progressive context growth: warm up at 4K, then 8K, then 16K over the run
```

### Data Setup

```bash
# Long-document public data (free): PG-19 (Project Gutenberg books) or
# a long-context subset of WikiText-103 concatenated into longer windows.
scripts/phase16_prepare_longdoc.sh data
```

### Tests

```rust
#[test]
fn rope_scaling_none_matches_v1_output_exactly() {
    // rope_scaling = None must be bit-identical to Phase 2's apply_rope().
}

#[test]
fn yarn_frequencies_interpolate_correctly_at_boundary() {
    // At position 0 and at original_max_seq_len, scaled freq ≈ unscaled freq.
}

#[test]
fn kv_cache_preallocates_to_scaled_max_seq_len() {
    // Cache buffer size matches the configured long-context length, not the
    // original model config's max_seq_len.
}

#[test]
fn attention_mask_handles_positions_beyond_original_max_seq_len() {}
```

### Milestone
```
Code/config support for Medium/Large continued pretraining with rope_scaling is
implemented. Actual 8K/16K checkpoint quality validation remains a user-run
training task and Phase 17 eval-harness task; no pretrained checkpoint is
released in this phase.

git commit -m "feat: Phase 16 — YaRN/NTK RoPE scaling, 16K+ context"
git tag v2.0.0-alpha.1
```

---

## Phase 17 — Evaluation Harness

**Duration:** 7–10 days | **Hardware:** i3 (small sets) + Kaggle (full sets)

### Goal
`aarambh-ai eval` command that runs perplexity-on-holdout plus free public
benchmark subsets, and prints a scorecard. This is the measuring stick for
every phase that follows — no model change after this point ships without a
before/after number.

### Tasks

**New crate `aarambh-ai-eval`:**
```
[x] src/ppl.rs
      compute_ppl(model, tokenizer, holdout_path) -> f32
      Reuses existing masked cross-entropy from aarambh-ai-train, eval-only
      (no gradient, no optimizer state)

[x] src/harness.rs
      EvalTask trait { name(), run(context, config) -> TaskScore }
      run_all(context, config) -> Scorecard

[x] src/tasks/mmlu_lite.rs
      Small free subset of MMLU (public, HuggingFace `cais/mmlu`)
      Multiple-choice via next-token logprob comparison over A/B/C/D

[x] src/tasks/hellaswag.rs
      Public HellaSwag validation subset
      Multiple-choice completion scoring via logprob ranking

[x] src/tasks/gsm8k_subset.rs
      Small free subset of GSM8K, exact-match on final numeric answer
      Reuses MathVerifier from aarambh-ai-finetune (Phase 10's GRPO verifier)

[x] src/tasks/humaneval_lite.rs
      Small free subset of HumanEval, pass@1 via sandboxed execution
      Adds and reuses CodeVerifier from aarambh-ai-finetune

[x] src/report.rs
      Scorecard { ppl, mmlu, hellaswag, gsm8k, humaneval, context_len_used }
      to_markdown() — writes a comparable table for CHANGELOG-style tracking
      to_json() — machine-readable for CI regression checks
```

**CLI (`aarambh-ai`):**
```
[x] aarambh-ai eval --config <cfg> --model <ckpt> --tasks ppl,mmlu,hellaswag
[x] aarambh-ai eval --config <cfg> --model <ckpt> --tasks all --out scorecard.json
[x] aarambh-ai eval --compare scorecard_before.json scorecard_after.json
```

### Data Setup

```bash
# All free, all public, no API key required.
scripts/phase17_prepare_eval_sets.sh data/eval
# Produces: data/eval/mmlu_lite/, hellaswag/, gsm8k_subset/, humaneval_lite/
```

### Tests

```rust
#[test]
fn ppl_on_known_holdout_matches_manual_calculation() {}

#[test]
fn mmlu_lite_scoring_picks_highest_logprob_option() {}

#[test]
fn gsm8k_subset_reuses_math_verifier_exact_match() {}

#[test]
fn scorecard_json_roundtrips_and_compare_reports_deltas() {}
```

### Milestone
```
`aarambh-ai eval` code/config support is implemented with offline unit tests
and fixture-safe CI. Full Tiny/Small/Medium benchmark scorecards are user-run
because this repository does not ship pretrained checkpoints.

git commit -m "feat: Phase 17 — evaluation harness (PPL, MMLU-lite, HellaSwag, GSM8K, HumanEval-lite)"
git tag v2.0.0-alpha.2
```

---

## Phase 18 — DoRA (Weight-Decomposed Low-Rank Adaptation)

**Duration:** 7–10 days | **Hardware:** i3 (small scale) + Kaggle (Small+)

### Goal
`aarambh-ai finetune dora` alongside the existing `lora`/`qlora`/`sft`/`merge`
commands. DoRA decomposes each adapted weight matrix into a magnitude vector
and a direction matrix, applies the LoRA-style low-rank update to direction
only, and trains magnitude as a separate free parameter. Same adapter
save/merge pattern as LoRA — this is additive, not a replacement.

### Tasks

**`aarambh-ai-finetune`:**
```
[x] src/dora.rs
      DoraConfig { rank, alpha, dropout, target_modules, group_size }
      DoraLinear:
        magnitude: Tensor        // ||W_0|| per output row, trainable
        direction_lora_a/b: Tensor  // low-rank update to direction, trainable
        base: Tensor              // frozen base weight (same as LoRA)
      forward:
        direction = (base + lora_b @ lora_a * scale) / row_norm(base + lora_b @ lora_a * scale)
        out = x @ (magnitude * direction).T
      merge: W_merged = magnitude * normalize(base + lora_b @ lora_a * scale)

[x] src/adapter.rs
      Extend adapter_config.json with `method: "lora" | "dora"`
      adapter.safetensors stores magnitude + direction_lora_a/b for DoRA

[x] src/dora.rs
      DoraAarambhModel mirrors LoraAarambhModel's forward path
      Same target-module suffix matching (attn.wq/wk/wv/wo, ffn gate/up/down)
      QDoRA variant: frozen base as PackedInt4Tensor, same as QLoRA

[x] src/trainer.rs
      DoraTrainer reuses SftTrainer's loop, swaps the adapter type
      run_dora_from_config, merge_dora_from_paths

[x] CLI
      aarambh-ai finetune dora --config <cfg> --data <sft.jsonl>
      aarambh-ai finetune qdora --config <cfg> --data <sft.jsonl>
      aarambh-ai finetune merge --adapter <path> --method dora
```

### Tests

```rust
#[test]
fn zero_direction_update_matches_normalized_base_forward() {
    // With lora_b = 0, direction = normalize(base), output = magnitude * that.
}

#[test]
fn dora_trainable_params_include_magnitude_and_direction_lora() {
    // Slightly more trainable params than plain LoRA at the same rank
    // (magnitude vector is extra), but far fewer than full fine-tune.
}

#[test]
fn dora_merge_produces_valid_normal_checkpoint() {
    // Merged model.safetensors loads through the existing infer path unmodified.
}

#[test]
fn qdora_dequantises_packed_int4_base_before_forward() {}
```

### Milestone
```
DoRA/QDoRA code paths, adapter metadata, merge support, CLI help, and offline
unit tests are implemented. Full Small DoRA vs LoRA scorecards are user-run
with the Phase 17 eval harness because this repository does not ship
pretrained checkpoints.

git commit -m "feat: Phase 18 — DoRA and QDoRA fine-tuning"
git tag v2.0.0-alpha.3
```

---

## Phase 19 — Vision Encoder + Projector

**Duration:** 10–14 days | **Hardware:** Kaggle (T4/P100)

### Goal
New `aarambh-ai-vision` crate. A frozen, pretrained ViT-class image encoder
(CLIP-B/32 scale, ~86M params, loaded from public SafeTensors weights via
Candle — no PyTorch bindings, no ONNX Runtime, no Python FFI, consistent
with the existing Dependency Policy) plus a small trainable projector MLP
that maps image patch embeddings into the language model's `d_model` space.
No language model weights are touched in this phase — only the projector
trains.

### Tasks

**New crate `aarambh-ai-vision`:**
```
[x] src/encoder.rs
      VisionEncoderConfig { patch_size, image_size, vit_d_model, vit_layers,
                             vit_heads, num_patches }
      ClipVisionEncoder — standard ViT: patch embed, position embed,
        pre-norm transformer blocks, frozen after load
      load_pretrained(path: &Path) -> ClipVisionEncoder  // SafeTensors, Candle

[x] src/preprocess.rs
      Image → tensor: resize, center-crop, normalize to CLIP's mean/std
      Uses `image` crate (new dependency, decode-only, no external service)

[x] src/projector.rs
      ProjectorConfig { vit_d_model, llm_d_model, hidden_mult }
      Projector — 2-layer MLP (GELU), vit_d_model -> hidden -> llm_d_model
      Trainable; this is the ONLY trainable component in Phase 19
      forward(vit_patch_embeds) -> llm_token_embeds  // one per image patch

[x] src/fusion.rs
      interleave_image_tokens(text_tokens, image_tokens, image_placeholder_id)
      Splices projected image-patch embeddings into the token embedding
      sequence at the position of a reserved <image> special token, LLaVA-style

[x] src/lib.rs
      VisionModel { encoder: ClipVisionEncoder (frozen), projector: Projector (trainable) }
```

**`aarambh-ai-tokenizer`:**
```
[x] Reserve <image> and <image_end> special token IDs (alongside existing
    <think>/</think> reserved IDs from Phase 7)
[x] Extend special-token validation so v2 tokenizers keep every reserved ID stable
```

**`aarambh-ai-train`:**
```
[x] src/vision_projector.rs
      Stage-1 recipe: encoder frozen, LLM frozen, only Projector trains
      Loss: standard next-token cross-entropy on image-caption pairs, loss
      masked to caption tokens only (image tokens contribute no target loss)
```

**CLI:**
```
[x] aarambh-ai train --config configs/vision_projector_pretrain.toml
[x] aarambh-ai infer --config <cfg> --image path.jpg --prompt "What is this?"
```

### Data Setup

```bash
# COCO Captions (free, public): image-caption pairs for projector pretraining
scripts/phase19_prepare_coco_captions.sh data
# Pretrained CLIP-B/32 SafeTensors weights (free, public, e.g. OpenCLIP release)
scripts/phase19_download_clip_weights.sh data/vision
```

### Tests

```rust
#[test]
fn vision_encoder_output_shape_matches_num_patches_x_vit_d_model() {}

#[test]
fn projector_output_shape_matches_llm_d_model() {}

#[test]
fn image_token_interleave_preserves_text_token_order() {
    // Non-image tokens keep their original positions and IDs.
}

#[test]
fn encoder_weights_do_not_change_after_projector_training_step() {
    // Explicit check that only the projector's VarMap enters AdamW.
}
```

### Milestone
```
Projector-pretrained Small-scale VLM produces on-topic (if rough) image
descriptions for held-out COCO validation images. Eval harness (Phase 17)
gains an image-captioning smoke check. No language model weights outside
the projector have moved.

git commit -m "feat: Phase 19 — frozen ViT vision encoder + trainable projector"
git tag v2.0.0-alpha.4
```

---

## Phase 20 — Vision-Language Training (Instruction Tuning)

**Duration:** 10–14 days | **Hardware:** Kaggle (T4/P100)

### Goal
Full VQA-style instruction tuning on top of Phase 19's pretrained projector.
The LLM side is adapted with DoRA (Phase 18) rather than full fine-tuning,
keeping this affordable on free Kaggle quota. Model can answer open-ended
questions about an image, not just caption it.

### Tasks

**`aarambh-ai-vision`:**
```
[ ] src/instruct_data.rs
      VqaExample { image_path, question, answer, thinking: Option<String> }
      JSONL schema compatible with existing SFT loss-masking conventions
      from Phase 9 (build_loss_mask reused unmodified)
```

**`aarambh-ai-finetune`:**
```
[ ] src/vlm_dora.rs
      VlmDoraTrainer combines: frozen vision encoder, frozen-or-tunable
      projector (config flag), DoRA-adapted LLM attention/FFN
      Only DoRA adapter params + optionally projector params enter AdamW
```

**`aarambh-ai-inference`:**
```
[ ] Extend existing infer path: --image flag prepends vision tokens before
    the KV-cache prefill step, otherwise inference is unchanged
[ ] Thinking engine (Phase 7) composes normally: <think> budget applies to
    the text generated after image tokens, unchanged from text-only behavior
```

### Data Setup

```bash
# LLaVA-Instruct-150K (free, public): image + multi-turn instruction pairs
scripts/phase20_prepare_llava_instruct.sh data
```

### Tests

```rust
#[test]
fn vqa_loss_mask_zeros_image_and_question_tokens() {
    // Only answer tokens (and thinking tokens, if present) carry loss.
}

#[test]
fn vlm_dora_trainable_params_exclude_frozen_vision_encoder() {}

#[test]
fn thinking_mode_works_identically_with_and_without_image_prefix() {}
```

### Milestone
```
Eval harness (Phase 17) gains a VQA task using a small free benchmark subset
(e.g. a VQAv2 validation slice). Before/after scorecard vs Phase 19's
caption-only baseline shows a measurable accuracy gain on open-ended
questions. Manual spot-check: model correctly answers counting and color
questions on held-out images at better than chance.

git commit -m "feat: Phase 20 — VQA instruction tuning via DoRA-adapted VLM"
git tag v2.0.0-alpha.5
```

---

## Phase 21 — Vision-Aware Self-Learning

**Duration:** 7–10 days | **Hardware:** Kaggle only (explicitly not i3-capable)

> See `SELF_LEARNING_V2.md` for the full design. Summary here for roadmap
> sequencing purposes.

### Goal
Extend the existing self-learning loop (`aarambh-ai-selflearn`) to handle
image-grounded turns. Text-only self-learning continues to work unmodified
on the i3, exactly as it does today. Vision self-learning is gated to
Kaggle only because a forward pass through the frozen ViT encoder on every
turn pushes memory well past the i3's comfortable CPU-safe envelope.

### Tasks

**`aarambh-ai-selflearn`:**
```
[ ] src/replay_buffer.rs
      Extend ReplayEntry with `image_ref: Option<PathBuf>` (or cached
      embedding path, to avoid re-running the frozen encoder on every replay)
      Schema version bump: replay_buffer_v2.jsonl, v1 files load unchanged
      with image_ref = None

[ ] src/vision_verifier.rs
      Grounded verifier for checkable VQA task types only (counting,
      color, yes/no presence questions) — analogous to Phase 10/12's
      MathVerifier/CodeVerifier pattern, NOT open-ended self-critique
      Open-ended image description falls back to existing self-critique,
      with the same noise caveats already documented for text

[ ] src/online_grpo.rs
      Extend online GRPO scoring to accept an optional vision_verifier;
      when absent, behavior is identical to today's text-only path

[ ] src/gating.rs
      require_hardware(Hardware::Kaggle) guard on any self-learn session
      that includes image turns; CLI errors clearly on i3 rather than
      silently degrading or OOMing
```

**CLI:**
```
[ ] aarambh-ai selflearn start --mode vision   (Kaggle only, errors on i3 with
                                                 a clear message pointing here)
[ ] aarambh-ai selflearn stats --mode vision
```

### Tests

```rust
#[test]
fn v1_replay_buffer_loads_with_image_ref_defaulting_to_none() {}

#[test]
fn vision_verifier_scores_counting_questions_exactly() {}

#[test]
fn vision_self_learn_session_rejects_i3_hardware_flag() {}

#[test]
fn text_only_self_learn_path_is_unmodified_by_this_phase() {
    // Regression test: Phase 12's existing i3 CPU-safe test suite still
    // passes byte-for-byte with no vision code in the path.
}
```

### Milestone
```
Vision self-learning session on Kaggle shows measurable score-trend
improvement on checkable VQA question types after 200+ turns, mirroring
the existing text-only "turn 200-500" behavior documented in
SELF_LEARNING.md. Text-only self-learning on i3 is verified unchanged.

git commit -m "feat: Phase 21 — vision-aware self-learning (Kaggle-only)"
git tag v2.0.0-alpha.6
```

---

## Phase 22 — Mixture of Experts

**Duration:** 10–14 days | **Hardware:** Kaggle (P100/A100 recommended)

### Goal
Optional MoE FFN layer as an alternative to the dense SwiGLU FFN, with
top-k expert routing and a load-balancing auxiliary loss. Framed honestly
per the eval harness: at Small/Medium scale, MoE is primarily a systems
learning exercise — the eval harness will show whether it earns its keep
at this parameter count before it's treated as a default.

### Tasks

**`aarambh-ai-nn`:**
```
[ ] src/moe.rs
      MoeConfig { num_experts, top_k, expert_ffn_dim, aux_loss_weight }
      Router — linear gate producing per-token expert logits
      top_k_gating(logits) -> (expert_indices, expert_weights)  // softmax over top-k only
      MoeFfn — num_experts independent SwiGLU FFNs, dispatched by router
      load_balancing_loss() — encourages uniform expert utilization
        (standard switch-transformer-style auxiliary loss)

[ ] src/dispatch.rs
      Token-to-expert dispatch via gather/scatter (dense masked matmul
      implementation first — simplest correct version; a sparse/grouped
      implementation is an optional follow-up, not required for this phase)
```

**`aarambh-ai-model`:**
```
[ ] Model config gains `moe: Option<MoeConfig>` per-layer or every-N-layers
[ ] New MoE-scale config: configs/small_moe.toml (Small dense-equivalent
    active params, more total params via experts)
```

**`aarambh-ai-train`:**
```
[ ] Total loss = cross_entropy_loss + aux_loss_weight * load_balancing_loss
[ ] Training log line gains `expert_util=[...]` per-expert utilization stats
```

### Tests

```rust
#[test]
fn top_k_gating_selects_correct_number_of_experts_per_token() {}

#[test]
fn load_balancing_loss_is_zero_at_perfectly_uniform_routing() {}

#[test]
fn moe_ffn_output_shape_matches_dense_ffn_output_shape() {
    // Drop-in compatible at the block level.
}

#[test]
fn dense_moe_none_config_reproduces_v1_dense_ffn_exactly() {}
```

### Milestone
```
Small-MoE trained on the same data/steps budget as the Phase 13 dense Small
baseline. Eval harness scorecard comparison published: report the actual
delta (positive, negative, or negligible) honestly rather than assuming
MoE wins at this scale — the finding itself is the milestone.

git commit -m "feat: Phase 22 — Mixture of Experts FFN with load-balanced routing"
git tag v2.0.0-alpha.7
```

---

## Phase 23 — Multi-GPU Training

**Duration:** 7–10 days | **Hardware:** Kaggle 2×T4 (when available)

### Goal
Data-parallel training across 2 GPUs via Candle, with graceful single-GPU
fallback (Kaggle's 2×T4 sessions are not always available — this must not
break the existing single-GPU path from Phase 13).

### Tasks

**`aarambh-ai-train`:**
```
[ ] src/distributed.rs
      DistributedConfig { world_size, rank, backend: Nccl }
      init_process_group() — Candle's NCCL-based collective ops
      all_reduce_gradients() — gradient sync after backward, before optimizer step
      Data sharding: each rank sees a disjoint shard of the DataLoader

[ ] src/trainer.rs
      Trainer gains an optional DistributedConfig; when None, behavior is
      byte-identical to the existing single-GPU/CPU path
      Checkpoint save only from rank 0
```

### Data Setup

```bash
# No new dataset; reuses Phase 13's WikiText-103 pipeline, sharded across ranks.
```

### Tests

```rust
#[test]
fn single_gpu_path_is_unchanged_when_distributed_config_is_none() {}

#[test]
fn gradient_all_reduce_averages_correctly_across_simulated_ranks() {
    // CPU-simulated 2-rank test using mocked collective ops.
}

#[test]
fn data_sharding_produces_disjoint_non_overlapping_batches() {}
```

### Milestone
```
2×T4 Kaggle session trains Small at measurably higher tok/s than the
single-T4 Phase 13 baseline (target: >1.6× throughput, accounting for
NCCL overhead). Falls back cleanly to single-GPU when only one GPU is
allocated by Kaggle.

git commit -m "feat: Phase 23 — multi-GPU data-parallel training via NCCL"
git tag v2.0.0-alpha.8
```

---

## Phase 24 — DPO Preference Tuning

**Duration:** 7–10 days | **Hardware:** Kaggle

### Goal
Direct Preference Optimization as a second alignment path alongside the
existing GRPO (Phase 10). DPO trains directly on (chosen, rejected) response
pairs without needing a reward model or an RL loop — simpler to run than
GRPO for open-ended chat quality where GRPO's verifier-based scoring doesn't
apply well.

### Tasks

**`aarambh-ai-finetune`:**
```
[ ] src/dpo.rs
      DpoConfig { beta, reference_free: bool }
      DpoDataset — { prompt, chosen, rejected } triples, JSONL
      dpo_loss(policy_logps, ref_logps, chosen_mask, rejected_mask, beta)
        Standard DPO loss: -log sigmoid(beta * (chosen_logratio - rejected_logratio))
      Reference model: frozen copy of the pre-DPO checkpoint (or
      reference_free variant using the implicit reward reformulation)

[ ] src/trainer.rs
      DpoTrainer — can wrap a DoRA adapter (Phase 18) instead of full
      fine-tune, keeping this affordable on Kaggle T4
```

### Data Setup

```bash
# Free, public preference datasets:
scripts/phase24_prepare_hh_rlhf.sh data       # Anthropic HH-RLHF
scripts/phase24_prepare_ultrafeedback.sh data # UltraFeedback
```

### Tests

```rust
#[test]
fn dpo_loss_decreases_when_chosen_logprob_increases_relative_to_rejected() {}

#[test]
fn reference_free_variant_matches_standard_dpo_at_beta_limit() {}

#[test]
fn dpo_dora_trainer_only_updates_adapter_params() {}
```

### Milestone
```
Eval harness scorecard shows improved preference-alignment proxy (e.g.
win-rate on a held-out preference pair subset) after DPO vs the SFT-only
baseline. GRPO (Phase 10) remains the preferred path for verifiable
math/code tasks; DPO is documented as the preferred path for open-ended
chat quality.

git commit -m "feat: Phase 24 — DPO preference optimization"
git tag v2.0.0-alpha.9
```

---

## Phase 25 — Speculative Decoding

**Duration:** 5–7 days | **Hardware:** Kaggle

### Goal
Tiny (25M) as a draft model proposing multiple tokens ahead, Large (1.3B,
or Medium) as the verifier accepting/rejecting them, for faster inference
without quality loss. Validated against the eval harness to confirm output
distribution is unchanged, not just faster.

### Tasks

**`aarambh-ai-inference`:**
```
[ ] src/speculative.rs
      SpeculativeConfig { draft_model_path, target_model_path, num_draft_tokens }
      speculative_decode_step():
        1. Draft model greedily/sampled-generates K tokens ahead
        2. Target model verifies all K in a single forward pass
        3. Accept the longest correct prefix; resample the first rejected
           token from the target model's corrected distribution
      Same tokenizer required for draft and target (shared vocab assumption
      documented explicitly)

[ ] CLI
      aarambh-ai infer --config <cfg> --speculative --draft-model <tiny.safetensors>
```

### Tests

```rust
#[test]
fn speculative_decoding_output_distribution_matches_target_only_decoding() {
    // Statistical equivalence test: same sampling distribution, not just
    // same greedy output — speculative decoding must not change quality.
}

#[test]
fn full_draft_rejection_falls_back_to_single_target_token_correctly() {}

#[test]
fn mismatched_tokenizer_vocab_between_draft_and_target_errors_clearly() {}
```

### Milestone
```
Measured wall-clock speedup on Kaggle T4 for Large-target/Tiny-draft
generation, target ≥1.8× tokens/second vs target-only decoding, with
eval harness scorecard confirming no quality regression (output
distribution equivalence, not just faster garbage).

git commit -m "feat: Phase 25 — speculative decoding (Tiny draft / Large target)"
git tag v2.0.0-alpha.10
```

---

## Phase 26 — Tool Use / Function Calling

**Duration:** 7–10 days | **Hardware:** i3 (inference) + Kaggle (SFT training)

### Goal
Model can emit structured tool calls matching a JSON schema, constrained at
decode time so output is always valid JSON — not just "usually" valid from
SFT alone.

### Tasks

**`aarambh-ai-inference`:**
```
[ ] src/grammar.rs
      JsonSchemaGrammar — compiles a JSON Schema into a token-level
      constraint (valid-next-token mask) applied during sampling
      Reuses the existing KV-cache decode loop; only the sampling step gains
      a mask derived from current grammar state

[ ] src/tool_calling.rs
      ToolDefinition { name, description, parameters: JsonSchema }
      ToolCallController wraps ThinkingController (Phase 7) — tool-call
      decoding composes with thinking budgets: model may think, then either
      answer directly or emit a constrained tool_call block
```

**`aarambh-ai-finetune`:**
```
[ ] src/tool_sft.rs
      Tool-call JSONL schema: {"instruction", "tools": [...], "tool_call": {...} | null, "response"}
      Loss masking extends Phase 9's build_loss_mask to tool_call token spans
```

**CLI:**
```
[ ] aarambh-ai infer --config <cfg> --tools tools.json --prompt "..."
```

### Tests

```rust
#[test]
fn grammar_constrained_decode_never_produces_invalid_json() {
    // Fuzz test: N random prompts, all outputs schema-valid.
}

#[test]
fn tool_call_composes_with_thinking_budget_without_conflict() {}

#[test]
fn no_tool_needed_case_falls_back_to_normal_generation() {}
```

### Milestone
```
Model correctly selects and formats tool calls for a held-out set of
multi-tool prompts (free/public function-calling eval subset, e.g. a
Gorilla or ToolBench sample), 100% JSON-schema-valid by construction,
correct-tool-selected rate tracked via the eval harness.

git commit -m "feat: Phase 26 — grammar-constrained tool use / function calling"
git tag v2.0.0-alpha.11
```

---

## Phase 27 — Inference Server

**Duration:** 10–14 days | **Hardware:** i3 (server runs fine on CPU for Tiny/Small)

### Goal
Rust HTTP/SSE server exposing an OpenAI-compatible chat completions API,
with continuous batching across concurrent requests. Serves your own local
checkpoints for your own testing and content — **no weights are published
by this phase**, same "source, not artifacts" policy as the rest of v2.0.

### Tasks

**New crate `aarambh-ai-serve`:**
```
[ ] src/server.rs
      Axum-equivalent routing (or reuse your own Ignitia/Arvik framework —
      natural fit here, keeps everything in the AarambhDevHub ecosystem)
      POST /v1/chat/completions  — OpenAI-compatible request/response shape
      POST /v1/completions
      GET  /v1/models             — lists locally loaded checkpoints only
      SSE streaming for `stream: true` requests

[ ] src/batching.rs
      ContinuousBatcher — dynamically merges concurrent decode steps into
      shared forward passes; new requests join at the next available step
      rather than waiting for a full batch to complete (standard continuous
      batching, not static batching)

[ ] src/session.rs
      Reuses aarambh-ai-inference's KV cache path per-request; safety layer
      (Phase 11) applies to every request by default, same as CLI infer
```

**CLI:**
```
[ ] aarambh-ai serve --config <cfg> --model <ckpt> --port 8080
[ ] aarambh-ai serve --config <cfg> --model <ckpt> --tools tools.json --thinking medium
```

### Tests

```rust
#[test]
fn chat_completions_response_matches_openai_schema() {}

#[test]
fn continuous_batching_two_concurrent_requests_do_not_cross_contaminate_kv_cache() {}

#[test]
fn safety_layer_applies_to_every_request_by_default() {}

#[test]
fn sse_stream_terminates_on_stop_token_or_max_tokens() {}
```

### Milestone
```
Server handles N concurrent local requests (N scaled to your i3's 8 GB RAM
and Tiny/Small checkpoint size) with continuous batching measurably
outperforming one-request-at-a-time throughput. No model weights leave
the machine; this is a local/self-hosted server, not a public deployment.

git commit -m "feat: Phase 27 — OpenAI-compatible inference server with continuous batching"
git tag v2.0.0-alpha.12
```

---

## Phase 28 — crates.io Publish (v2.0.0 Release)

**Duration:** 5–7 days | **Hardware:** all

### Goal
Publish all library crates to crates.io under 2.0.0, exactly mirroring
RELEASE.md's v1.0.0 discipline: **source code only, zero model artifacts.**
No pretrained checkpoints, adapters, or GGUF files are attached to this
release either — same policy, one major version later.

### Tasks

```
[ ] Package manifests: all 16 library crates + CLI set to version 2.0.0
[ ] Flip `publish = false` → real crates.io metadata (description, license,
    repository, keywords) for crates that are ready; keep publish = false
    for any crate you judge still too unstable for semver commitments
[ ] cargo publish --dry-run for every publishable crate, in dependency-layer
    order (core → tokenizer/data/nn → kernel/model/weights/quant/vision →
    train/finetune → inference/safety/selflearn → serve → binary)
[ ] CHANGELOG.md: v2.0.0 entry summarizing Phases 16–27
[ ] README.md, ARCHITECTURE.md, ROADMAP.md: merge in the v2 additions (or
    keep _V2 docs as addenda, your call)
[ ] RELEASE.md: v2.0.0 checklist, explicitly restating "no pretrained
    checkpoints, no model artifacts" for this release line too
[ ] .github/release-notes/v2.0.0.md
[ ] CI: extend existing workflow to cover the 3 new crates + eval harness
    smoke run
```

### Milestone
```
cargo install --path aarambh-ai
aarambh-ai --version  → aarambh-ai 2.0.0
git tag v2.0.0
git push origin v2.0.0

Crates resolve from crates.io for any published subset; unpublished crates
remain source-only exactly as documented. No model weights attached to the
GitHub Release.

git commit -m "chore: v2.0.0 — crates.io publish, source release"
```

---

## Complete Phase Summary

| # | Phase | Key Deliverable | Hardware | Duration |
|---|---|---|---|---|
| 16 | Long Context | YaRN/NTK RoPE scaling, 4K → 16K+ | Kaggle | 7–10 days |
| 17 | Evaluation Harness | `aarambh-ai eval`, PPL + MMLU-lite/HellaSwag/GSM8K/HumanEval-lite | i3 + Kaggle | 7–10 days |
| 18 | DoRA | Weight-decomposed LoRA in `aarambh-ai-finetune` | i3 + Kaggle | 7–10 days |
| 19 | Vision Encoder + Projector | ✅ New `aarambh-ai-vision` crate, frozen ViT + trainable projector | Kaggle | 10–14 days |
| 20 | Vision-Language Training | VQA instruction tuning via DoRA-adapted VLM | Kaggle | 10–14 days |
| 21 | Vision-Aware Self-Learning | Image-grounded replay + verifier, Kaggle-only | Kaggle only | 7–10 days |
| 22 | Mixture of Experts | Top-k router, load-balancing loss | Kaggle | 10–14 days |
| 23 | Multi-GPU Training | Data-parallel training via NCCL | Kaggle 2×T4 | 7–10 days |
| 24 | DPO Preference Tuning | Preference optimization alongside GRPO | Kaggle | 7–10 days |
| 25 | Speculative Decoding | Tiny-draft / Large-target speedup | Kaggle | 5–7 days |
| 26 | Tool Use / Function Calling | Grammar-constrained JSON tool calls | i3 + Kaggle | 7–10 days |
| 27 | Inference Server | OpenAI-compatible server, continuous batching, local-only | i3 | 10–14 days |
| 28 | crates.io Publish | Source-only 2.0.0 release, zero model artifacts | all | 5–7 days |

**Total realistic estimate: 99–140 days (~3.3–4.7 months)**

---

## Dependency Policy Additions (v2.0)

| Dependency | Allowed crates | Reason |
|---|---|---|
| `image` | vision | Image decode/resize/normalize only, no external service calls |
| NCCL bindings (via `candle-core` CUDA features) | train | Multi-GPU collective ops |
| (server framework — Ignitia/Arvik or equivalent) | serve | HTTP/SSE routing |

**Still forbidden everywhere:** PyTorch bindings (`tch-rs`), ONNX Runtime
(`ort`), Python FFI, `llama.cpp` as a backend. All computation goes through
`candle`, including the new vision encoder. Pretrained CLIP weights are
loaded as SafeTensors — never as a PyTorch checkpoint requiring `tch-rs`.

**Version policy:** unchanged from v1 — pin major versions, test the whole
workspace on any `candle-core` upgrade.

---

## What's Explicitly Out of Scope for v2.0

- Releasing any pretrained checkpoint, adapter, or GGUF file
- Multi-node distributed training (Phase 23 is 2-GPU single-node only)
- Public/hosted deployment of the inference server (Phase 27 is local-only)
- Video or audio modalities (vision is images only)
- Synthetic API-generated training data (all datasets in this roadmap are
  free and public; paid synthetic data remains an optional future upgrade,
  not a requirement of any v2.0 phase)

These are natural v3 candidates once budget and a released model exist.
