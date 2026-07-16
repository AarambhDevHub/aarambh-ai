# ARCHITECTURE_V3.md — aarambh-ai v3.0

> Companion to `ARCHITECTURE.md` and `ARCHITECTURE_V2.md`. This document
> covers **only what v3.0 adds or changes** — sections here are numbered to
> continue directly from v2's Section 35. Anything not mentioned here
> (tokenizer core, RMSNorm, GQA/RoPE baseline, SwiGLU, quantisation
> baseline, safety layer, vision encoder baseline, etc.) is unchanged from
> v1.0.0/v2.0.0. Read `ARCHITECTURE.md` and `ARCHITECTURE_V2.md` first;
> this is the delta on top of both.

---

## Table of Contents

36. [What's New in v3.0](#36-whats-new-in-v30)
37. [Updated Workspace — 18 Library Crates](#37-updated-workspace--18-library-crates)
38. [Gated DeltaNet: Hybrid Linear Attention](#38-gated-deltanet-hybrid-linear-attention)
39. [DeepSeek Sparse Attention (DSA)](#39-deepseek-sparse-attention-dsa)
40. [Fine-Grained MoE with Shared Expert](#40-fine-grained-moe-with-shared-expert)
41. [Multi-Token Prediction (MTP)](#41-multi-token-prediction-mtp)
42. [On-Policy Distillation](#42-on-policy-distillation)
43. [Native QAT](#43-native-qat)
44. [Video Understanding](#44-video-understanding)
45. [Document Understanding](#45-document-understanding)
46. [Long-Horizon Tool-Use Chains](#46-long-horizon-tool-use-chains)
47. [Forgetting Diagnostics](#47-forgetting-diagnostics)
48. [Max Thinking Mode](#48-max-thinking-mode)
49. [Updated Dependency Layers](#49-updated-dependency-layers)
50. [Updated Memory & Compute Estimates](#50-updated-memory--compute-estimates)
51. [Updated Hardware Strategy](#51-updated-hardware-strategy)
52. [What's Explicitly Out of Scope](#52-whats-explicitly-out-of-scope)

---

## 36. What's New in v3.0

v2.0 widened the pipeline into a new modality (vision) and deepened
training infrastructure (MoE, multi-GPU). v3.0 pushes on three more
fronts, directly informed by the 2026 open-weight model landscape (GLM,
Qwen, MiniMax, DeepSeek-derived architectures):

1. **A new attention stack** — Gated DeltaNet hybrid linear attention plus
   DeepSeek Sparse Attention replace v1/v2's all-full-attention design as
   the default, cutting KV-cache pressure and improving long-context
   throughput, retrofitted onto existing checkpoints via continued
   pretraining rather than requiring a from-scratch rebuild.
2. **A more efficient training regime** — fine-grained MoE routing with a
   shared expert (upgrading v2's dense MoE), multi-token prediction heads,
   on-policy distillation, and native quantization-aware training, all
   aimed at getting more capability per active parameter and per training
   token.
3. **Wider still — two more modalities** — native video and document
   understanding, both sharing v2's frozen-ViT-encoder-plus-trainable-
   projector economics, built in from this phase forward rather than
   adapted on later.
4. **More capable agents** — long-horizon tool-use chains upgrade v2's
   single-call tool use into sustained multi-step reasoning over tool
   results, including multimodal ones.
5. **A measurement layer for memory itself** — forgetting diagnostics give
   both aarambh-ai's self-learning loop and the separate Manas project a
   shared, measured signal for catastrophic forgetting, rather than only
   inferring it from score regressions after the fact.
6. **A fifth thinking mode** — Max, sitting above High in the existing
   None/Low/Medium/High token-budget lineup, for tasks that genuinely
   need more room to reason than 4,096 tokens allows.

**Still true in v3.0, unchanged from v1/v2:** no pretrained checkpoints,
adapters, or GGUF files are released. This is a source/engineering
release. See `RELEASE.md`.

---

## 37. Updated Workspace — 18 Library Crates

Two new crates. Everything else is extended in place — no crate is
removed or renamed, matching v2's discipline.

```
aarambh-ai/
├── Cargo.toml
├── ARCHITECTURE.md / ARCHITECTURE_V2.md / ARCHITECTURE_V3.md
├── ROADMAP.md / ROADMAP_V2.md / ROADMAP_V3.md
├── SELF_LEARNING.md / SELF_LEARNING_V2.md / SELF_LEARNING_V3.md
│
├── crates/
│   │   ...Layers 0–6 from v1.0.0/v2.0.0, extended (see §38–47 below)...
│   │
│   ├── aarambh-ai-distill/           ← NEW, LAYER 5: On-policy distillation
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── rollout.rs            ← student on-policy generation
│   │       ├── teacher_score.rs      ← TeacherScorer trait, local/dataset backends
│   │       └── distill_loss.rs       ← on-policy distillation objective
│   │
│   └── aarambh-ai-agent/             ← NEW, LAYER 5: Multi-step tool-use chains
│       └── src/
│           ├── lib.rs
│           ├── chain.rs              ← ToolChain orchestrator, stopping conditions
│           ├── result_ingestion.rs   ← ToolResult (text/image/video/document)
│           └── state.rs              ← ChainState, eviction/summarisation policy
│
└── aarambh-ai/                       ← LAYER 6: CLI binary
    └── src/cmd/
        ├── ...train.rs / infer.rs / finetune.rs / quantise.rs / convert.rs / eval.rs / serve.rs...
        └── agent.rs                  ← NEW: `aarambh-ai agent`
```

### Extended (not new) crates in v3.0

| Crate | v3.0 additions |
|---|---|
| `aarambh-ai-nn` | `gated_deltanet.rs` (§38), `sparse_attention.rs` (§39), `moe.rs`/`dispatch.rs` extended (§40), `mtp.rs` (§41) |
| `aarambh-ai-model` | `attention_schedule: Option<HybridAttentionSchedule>`, `dsa_config: Option<DsaConfig>`, extended `MoeConfig`, `mtp: Option<MtpConfig>` on model config |
| `aarambh-ai-train` | Hybrid-attention retrofit recipe, indexer training (DSA), router warm-start (MoE), `mtp_loss.rs`, distillation training loop, QAT recipe, Max-mode GRPO re-run on High-insufficient problems |
| `aarambh-ai-weights` | Partial-checkpoint loading (load some layers, fresh-init others) for attention retrofit |
| `aarambh-ai-quant` | `qat.rs` — `FakeQuantize`, `QatConfig`, straight-through estimator |
| `aarambh-ai-vision` | `video_sample.rs`, `temporal_fusion.rs` (§44), `document_sample.rs`, `layout_projector.rs` (§45) |
| `aarambh-ai-tokenizer` | `<video>`/`<video_end>`/`<frame_sep>` and `<document>`/`<document_end>`/`<page_sep>` reserved tokens |
| `aarambh-ai-finetune` | Multi-frame/multi-page VLM tuning (extends v2's `vlm_dora.rs`), multi-step tool-use SFT loss masking |
| `aarambh-ai-inference` | MTP heads reusable as speculative-decode draft source; `thinking.rs` gains the `Max(16384)` `ThinkingMode` variant (§48) |
| `aarambh-ai-eval` | `forgetting.rs` (§47), new video/document/multi-step-tool-use task subsets |
| `aarambh-ai-selflearn` | Forgetting-diagnostics wiring, shared export format for Manas — see `SELF_LEARNING_V3.md` |

### Updated Crate Count

```
v1.0.0: 14 crates (13 library + 1 binary)
v2.0.0: 17 crates (16 library + 1 binary)
v3.0.0: 19 crates (18 library + 1 binary)
```

---

## 38. Gated DeltaNet: Hybrid Linear Attention

v1/v2's attention is full causal attention with GQA and YaRN/NTK RoPE
scaling (`ARCHITECTURE.md` §6.3, `ARCHITECTURE_V2.md` §21) at **every**
layer. Full attention's compute cost is quadratic in sequence length, and
its KV cache grows linearly with tokens generated — both become the
dominant cost at long context, which is exactly what v2 Phase 16 needed
RoPE scaling to even reach usefully. Gated DeltaNet attacks the problem
from a different angle: instead of scaling the same mechanism further, it
replaces most layers with a different mechanism whose cost profile is
flat in sequence length.

### 38.1 The recurrence

Each Gated DeltaNet layer maintains a fixed-size matrix-valued recurrent
state `S` (shape `[d_k, d_v]` per head) that is updated at every token
rather than an ever-growing set of key/value pairs:

```
At each token t, for each head:
  q_t, k_t         ← L2-normalize(SiLU(depthwise_conv(proj(x_t))))
  v_t              ← SiLU(depthwise_conv(v_proj(x_t)))
  β_t              ← sigmoid(beta_proj(x_t)), one scalar per head
  α_t              ← exp(-exp(A_log) · softplus(alpha_proj(x_t) + dt_bias)),
                    one stable FP32 decay scalar per head

  # Delta rule: state moves toward better predicting v_t from k_t,
  # rather than a plain running-sum update
  S_decay = α_t · S_{t-1}
  error_t = v_t − k_tᵀ S_decay
  S_t     = S_decay + β_t · k_t ⊗ error_t

  output_t = out_proj(SiLU(gate_proj(x_t)) · RMSNorm(q_tᵀ S_t))
```

`α_t` (gate) controls how much of the previous state survives —
this is the "Gated" in Gated DeltaNet, and is what lets the layer
selectively forget stale information rather than accumulating it
forever, unlike a plain linear-attention running sum. `β_t` controls how
strongly the current token's key/value pair overwrites the state's
existing prediction for that key — the delta-rule correction term
`(v_t − S_{t-1}ᵀ k_t)` is zero when the state already predicts `v_t`
correctly from `k_t`, so well-predicted tokens barely perturb the state.

### 38.2 Two forward forms

```rust
pub enum DeltaNetForm {
    /// O(1) state per step, used during autoregressive generation.
    /// Processes one token at a time; no KV cache growth.
    Sequential,

    /// Chunk-bounded differentiable form used during training. Splits the
    /// sequence into fixed execution chunks (chunk_size, default 64) while
    /// preserving the exact token dependency of the delta rule. It uses
    /// Candle operations so backward reaches every recurrent parameter.
    ChunkParallel { chunk_size: usize },
}
```

Autoregressive prefill/decode dispatches each recurrent update to the
CPU-parallel or CUDA kernel when its layout and dtype are supported. Training
uses the differentiable Candle recurrence because the custom decode operation
is intentionally inference-only; unsupported devices and layouts retain the
same numerical fallback rather than changing the model rule.

`DeltaNetState` (the recurrent state struct) replaces `KvCache` for
Gated DeltaNet layers specifically — it is a fixed-size tensor allocated
once per generation session, updated in place, never reallocated as
generation continues. `HybridKvCache` (extending v1/v2's `KvCache`)
becomes a per-layer enum: `Full(KvCache)` for full-attention layers,
`Linear(DeltaNetState)` for Gated DeltaNet layers.

### 38.3 Hybrid scheduling

```rust
pub struct HybridAttentionSchedule {
    /// Every Nth layer (0-indexed) is full attention; the rest are
    /// Gated DeltaNet. Default N = 4 (1 in 4 layers full attention),
    /// following the field's reported ratio for maintaining precise
    /// long-range recall while keeping most layers linear-cost.
    pub full_attention_every_n: usize,
    /// Recurrent head dimensions, convolution width and execution chunk size.
    pub gated_deltanet: GatedDeltaNetConfig,
}

impl HybridAttentionSchedule {
    pub fn kind_for_layer(&self, layer_idx: usize) -> AttentionKind {
        if layer_idx % self.full_attention_every_n == 0 {
            AttentionKind::Full
        } else {
            AttentionKind::GatedDeltaNet
        }
    }
}
```

This is a genuinely a per-layer-index decision, not a global switch — a
12-layer Medium model with `full_attention_every_n = 4` runs layers
`0, 4, 8` as full attention and the remaining 9 layers as Gated
DeltaNet. `attention_schedule: None` on the model config reproduces
v1/v2's all-full-attention behaviour exactly — this is enforced by a
byte-identical-output regression test, not just documented as intent.

### 38.4 Retrofit procedure (not a from-scratch rebuild)

```
1. Load an existing v2 checkpoint's SafeTensors file
2. For each layer index scheduled as Full by HybridAttentionSchedule:
     load that layer's attention weights unchanged (q/k/v/o projections,
     RoPE buffers) — byte-identical to the source checkpoint
3. For each layer index scheduled as GatedDeltaNet:
     discard the source checkpoint's attention weights for that layer
     entirely; fresh-initialise DeltaNet's q/k/v projections and the
     α_t/β_t gate projections (small, Xavier-uniform init)
4. Continue pretraining at a reduced learning rate (default 0.1x the
   original pretraining LR) so the untouched Full-attention layers and
   the FFN/embedding weights do not drift far from their already-trained
   state while the fresh GatedDeltaNet layers learn from near-random init
5. Validate: eval-harness score (`ARCHITECTURE_V2.md` §17 / `aarambh-ai-eval`)
   on the same holdout before and after retrofit; regression must stay
   within a documented tolerance band (default: no more than 2 percentage
   points absolute on any tracked task) before the retrofit is accepted
```

Checkpoint tensor names make the split explicit and greppable:
`blocks.N.attn.{wq,wk,wv,wo}.weight` for Full-attention layers (unchanged
naming from v1/v2), versus `blocks.N.deltanet.{q,k,v}_proj.weight`,
`blocks.N.deltanet.{alpha,beta,gate,out}_proj.weight`, causal convolution
weights, `A_log`, `dt_bias`, and output norm for Gated DeltaNet layers — a
config's attention schedule can be reconstructed just by reading which
tensor names are present for each layer index, without needing to parse
the TOML config alongside the checkpoint.

**Division of labour going forward:** Full attention stays the
mechanism of record for precise long-range recall (needle-in-a-haystack-
style retrieval, exact copying from far back in context). Gated DeltaNet
is the mechanism of record for everything else — the bulk of ordinary
next-token prediction, where a compressed running summary of the past is
sufficient. This mirrors exactly how v1/v2's SwiGLU FFN and MoE FFN
already coexist as drop-in alternatives selected per-layer (§40) — Gated
DeltaNet is architecturally the same kind of decision, one level up, at
the attention block instead of the FFN block.

---

## 39. DeepSeek Sparse Attention (DSA)

The layers §38 still schedules as full attention pay full quadratic
cost. DSA reduces that cost for exactly those remaining layers, without
touching Gated DeltaNet layers at all (they have no quadratic cost to
reduce in the first place).

### 39.1 Two-stage attention

```
Stage 1 — Lightning indexer (cheap):
  For query position t, score every preceding key BLOCK (not every
  individual key) using a small auxiliary projection, much lower-rank
  than the full attention score computation:

    index_score(t, block_b) = lightning_proj(q_t) · lightning_proj(k̄_b)

  where k̄_b is a pooled (mean) representation of block b's keys.
  Cost: O(seq_len / block_size) per query, versus O(seq_len) for the
  real thing — this is what makes indexing cheap enough to run on
  every query without itself becoming the bottleneck.

Stage 2 — Sparse full attention (selected blocks only):
  Select the top_k highest-scoring blocks for query t (default top_k=16,
  block_size=64 → attends over at most 1,024 keys regardless of true
  sequence length)
  Run ordinary scaled-dot-product attention (unchanged math from
  `ARCHITECTURE.md` §6.3) restricted to keys within the selected blocks
```

```rust
pub struct DsaConfig {
    pub block_size: usize,       // default 64
    pub top_k_blocks: usize,     // default 16
    pub min_seq_len_for_sparsity: usize, // default 2048 — below this,
                                          // fall back to plain full
                                          // attention transparently
}
```

### 39.2 Training the indexer

The lightning indexer is a real learned component, not a heuristic — if
its block rankings don't agree with what full attention would actually
have weighted highly, sparsity silently drops important context and
quality degrades without any visible error. Phase 30's training recipe:

```
On a configurable fraction of training steps (default: every 8th step):
  1. Run full (unsparsified) attention for the scheduled DSA layers,
     recording the true per-block attention mass (sum of softmax weight
     assigned to each block)
  2. Train the lightning indexer with a listwise ranking loss against
     that true per-block mass — the indexer's job is only to rank blocks
     correctly, not to reproduce exact attention weights
  3. On the remaining steps, run the cheap indexer + sparse attention
     path only, using the indexer as trained so far

This is a form of self-distillation: full attention teaches the cheap
indexer, on a sampled subset of steps, without ever needing an external
teacher model.
```

### 39.3 KV-cache accounting

Even with DSA, cache memory for these layers still grows with sequence
length — indexing selects *which* stored keys to attend to, it does not
avoid storing them in the first place. The payoff is entirely in
*compute* (fewer keys touched per attention op) plus a smaller
*constant* on cache memory access patterns, not a change to the
asymptotic memory-growth curve the way Gated DeltaNet (§38) achieves.
This distinction matters for capacity planning and is called out
explicitly in the memory table (§50) rather than left implicit.

Combined with §38, the default v3.0 attention stack per model is:
mostly Gated DeltaNet (linear, constant-memory), the scheduled minority
of layers running DSA-sparse full attention (reduced compute, still
linearly-growing cache), and zero fully-dense unsparsified full-attention
layers by default — plain dense full attention remains available as an
explicit config fallback (`dsa_config: None` on a Full-attention-scheduled
layer) for debugging and side-by-side comparison, not removed from the
codebase.

### 39.4 Phase 30 implementation contract

Phase 30 implements DSA at causal block granularity. `DsaAttention` preserves
the existing `blocks.N.attn.*` projection names and adds only
`blocks.N.dsa.index_q.weight` and `blocks.N.dsa.index_k.weight`, so a Phase 29
checkpoint retrofits without remapping its trained tensors. The index width is
the attention head width and receives the same RoPE/YaRN positions as GQA.

Completed index-key blocks are pooled in FP32. Selection is shared across all
GQA query heads, deterministic on ties, always contains the current partial
block, excludes future blocks and future tokens within the current block, and
returns chosen blocks in chronological order. Dense fallback is exact below
`min_seq_len_for_sparsity` or whenever the causal block count fits in
`top_k_blocks`.

`DsaKvCache` stores the ordinary full K/V tensors, one pooled index key per
completed block, and at most one active block of index keys. The selected K/V
working set is bounded by `top_k_blocks * block_size`, but total K/V cache
storage remains O(sequence length). Inference statistics therefore report
stored cache and selected working-set bytes as separate quantities.

Every eighth optimizer step by default computes true dense causal attention
mass aggregated by key block and query head. That detached distribution trains
only the low-rank indexer with listwise KL; ordinary steps train the model
through sparse attention. CPU inference uses a Rayon selected-mask online
softmax path. CUDA builds include F32/F16/BF16 top-k, selected-block forward,
and teacher-mass PTX kernels, with Candle fallbacks for unsupported builds.

---

## 40. Fine-Grained MoE with Shared Expert

v2 Phase 22 shipped a working MoE (`ARCHITECTURE_V2.md` §26): a top-k
softmax router selecting among a modest number of relatively large
experts, using dense-masked-matmul dispatch (every expert computes on
every token, then the result is masked/weighted by the router — the
batch-simple, GPU-memory-hungrier alternative to sparse/grouped dispatch,
a deliberate v2 trade-off that remains true in v3; genuine sparse
dispatch is still out of scope, see §52).

### 40.1 What changes: the shape of the expert pool, not the dispatch mechanism

```rust
pub struct MoeConfig {
    // ...v2's existing fields unchanged: num_experts, top_k,
    // aux_loss_weight, every_n_layers...

    /// v3 addition. Each v2-era expert is conceptually split into this
    /// many smaller experts, each with FFN hidden width divided by this
    /// factor, so total FFN parameter budget stays ~constant while the
    /// number of distinct routing targets increases. Default 1
    /// (no change from v2's coarse-grained MoE).
    pub fine_grained_factor: usize,

    /// v3 addition. Number of experts that are always active for every
    /// token, independent of router decisions. Default 0 (no change
    /// from v2). These are drawn from a separate, dedicated pool — not
    /// double-counted against num_experts.
    pub num_shared_experts: usize,
}
```

Setting `fine_grained_factor=1, num_shared_experts=0` reproduces v2's
Phase 22 MoE behaviour exactly — enforced by a byte-identical-output
regression test, the same discipline §38 applied to the attention
schedule default.

### 40.2 Fine-grained routing

With `num_experts=8, fine_grained_factor=4`, the router now selects
among `8 × 4 = 32` experts, each with FFN hidden width `d_ffn / 4`
relative to what a single v2-era expert would have used — total FFN
parameter count across the 32 fine-grained experts is approximately
equal to the original 8 coarse experts', but the router can now express
much more precise per-token specialisation (e.g. one fine-grained expert
handling "numeric formatting inside code" rather than one coarse expert
handling all of "code" broadly). This finer granularity is the mechanism
credited for results like MiniMax M2.7 reaching near-frontier scores
with only 10B activated parameters — routing to many small specialists
outperforms routing to a few large generalists at matched active-compute
budgets.

### 40.3 Shared expert path

```rust
pub struct SharedExpertPath {
    /// One SwiGLU FFN per shared expert (same structure as v1's
    /// dense FFN, ARCHITECTURE.md §6.6), always evaluated for every
    /// token — no gating, no top-k selection.
    experts: Vec<SwiGluFfn>,
}

impl SharedExpertPath {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Sum of all shared experts' outputs, added unconditionally
        // to the routed-expert output before the residual connection:
        //   output = routed_expert_output + shared_expert_output
        let mut out = self.experts[0].forward(x)?;
        for expert in &self.experts[1..] {
            out = (out + expert.forward(x)?)?;
        }
        Ok(out)
    }
}
```

The shared expert path guarantees every token a baseline FFN
transformation regardless of what the router decides, which catches
general-purpose computation that shouldn't be routing-dependent in the
first place (e.g. basic positional/syntactic processing useful for every
token type). It is explicitly excluded from the load-balancing
auxiliary loss (`ARCHITECTURE_V2.md` §26's `aux_loss_weight` term) —
since it is never a routing decision, penalising its "utilisation" would
be meaningless; the load-balancing loss continues to apply only to the
routed, fine-grained expert pool.

Checkpoint tensor names extend v2's convention directly:
`blocks.N.ffn.experts.E.{w_gate,w_up,w_down}.weight` for routed
fine-grained experts (E now ranging over `num_experts × fine_grained_factor`
rather than v2's `num_experts`), and
`blocks.N.ffn.shared_experts.E.{w_gate,w_up,w_down}.weight` for the new
shared pool — kept as a distinct tensor-name namespace so a checkpoint
loader can distinguish routed from shared weights without needing the
TOML config alongside it.

### 40.4 Router warm-start

Rather than training the finer-grained router from scratch, Phase 31
initialises it from the already-trained v2 coarse router's weights where
dimensions allow — the coarse router's learned "this token needs
math-like processing" signal is a reasonable starting point for the
finer-grained router even though the finer router now needs to also
decide *which* math-like expert among several. This is cheaper than
cold-start training and is consistent with the retrofit philosophy §38
established for attention.

### 40.5 Validation methodology

Phase 31 ships an expert-count sweep script that trains small runs
across a grid of `(num_experts, fine_grained_factor, top_k)` and logs
eval-harness deltas at matched *active*-parameter budgets, mirroring the
"does the trade pay off" discipline v2 Phase 22 already established for
MoE versus dense — fine-grained routing is adopted only where the sweep
shows it actually beats coarse routing at equal active-parameter cost,
not assumed from the literature alone.

### 40.6 Phase 31 implementation contract

`MoeConfig::routed_expert_count` computes `num_experts ×
fine_grained_factor`, while `fine_grained_expert_dim` requires the configured
coarse `expert_ffn_dim` to divide exactly by the factor. The model constructs
the router against that expanded pool and constructs every routed and shared
SwiGLU with the divided width. The default factor `1` and zero shared experts
produce exactly identical logits to the Phase 22 path.

Dense dispatch remains the execution contract: all routed experts still run
for every token. The implementation accumulates each weighted expert output
immediately instead of stacking every expert result, reducing temporary output
memory without pretending this is sparse dispatch. Shared outputs are summed
after routed accumulation and are absent from router utilization and auxiliary
loss statistics.

Coarse-to-fine SafeTensors retrofit is function-preserving at initialization.
Each coarse router row is repeated for its children; gate/up rows and down
columns are partitioned by child; every child down projection is multiplied by
the split factor so the expanded top-k weighted sum reconstructs the coarse
output. New shared gate/up tensors keep their initialized values and shared
down projections start at zero. Consequently, the shared path learns from the
first update without perturbing the source checkpoint's initial function.

The Medium and Large Phase 31 recipes use 8 coarse groups, factor 4, top-k 8,
and one shared expert. Their fine widths are 424 and 832 respectively. The
matched coarse source recipes use top-k 2, preserving routed active width when
the target top-k is multiplied by the factor. `scripts/phase31_sweep_moe.sh`
enforces scratch-only comparison runs and emits raw scorecards plus
baseline-relative Markdown; repository documentation does not substitute
unrun benchmark values for those artifacts.

---

## 41. Multi-Token Prediction (MTP)

v1/v2 train on a single objective at every position: predict the next
token. MTP adds `num_future_tokens` lightweight auxiliary heads, each
predicting one additional token further into the future, from the same
shared trunk hidden state — this mirrors the MTP-3-style setups (3
future-token heads) seen in GLM-4.7/MiniMax-M2.1-derived architectures.

### 41.1 Head structure

```rust
pub struct MtpConfig {
    /// How many future-token heads to attach. Default 2, following the
    /// "MTP-2" variant of the pattern (predicts t+1 [main head, always
    /// present] and t+2 [one auxiliary head]); MTP-3 (t+1, t+2, t+3) is
    /// a documented config option, not the default, since each
    /// additional head adds real training compute.
    pub num_future_tokens: usize,

    /// Weight of the auxiliary MTP loss relative to the main next-token
    /// loss. Default 0.3 — auxiliary signal, not the primary objective.
    pub aux_loss_weight: f64,
}

pub struct MtpHead {
    /// A lightweight additional transformer block (not a full stack —
    /// one block is sufficient since it only needs to refine, not
    /// recompute, the trunk's representation) plus a tied-or-separate
    /// LM head.
    refine_block: TransformerBlock,
    lm_head: Linear,
}

impl MtpHead {
    /// Predicts the token at position t + offset, given the trunk's
    /// hidden state at position t and the (real, during training;
    /// previously-predicted, during MTP-based speculative decoding)
    /// token embeddings for positions between t and t + offset.
    pub fn forward(&self, trunk_hidden: &Tensor, intervening_embeds: &Tensor) -> Result<Tensor> {
        let combined = concat_along_hidden_dim(trunk_hidden, intervening_embeds)?;
        let refined = self.refine_block.forward(&combined)?;
        self.lm_head.forward(&refined)
    }
}
```

### 41.2 Training loss

```
main_loss  = cross_entropy(main_lm_head_logits, next_token_targets)
mtp_loss_k = cross_entropy(mtp_head_k_logits, token_at_offset_k_targets)
              for k in 1..=num_future_tokens

total_loss = main_loss + aux_loss_weight × mean(mtp_loss_1, ..., mtp_loss_K)
```

The training scorecard (extending `ARCHITECTURE.md`'s existing loss
logging) reports `main_loss` and each `mtp_loss_k` separately, not just
the combined total — this makes it possible to see directly whether the
auxiliary heads are learning at all, independent of whether they're
actually helping the main objective.

### 41.3 Reuse as a speculative-decoding draft

v2 Phase 25 (`ARCHITECTURE_V2.md` §29) implements speculative decoding
using the Tiny (25M) model as a separate draft checkpoint proposing
tokens for the Large (1.3B) target to verify. When MTP is enabled, the
target model's own MTP heads already predict several tokens ahead from
the shared trunk — they can stand in as the draft source directly,
removing the need to load a second, separate checkpoint:

```
Without MTP (v2 Phase 25 path):
  Tiny model (separate checkpoint) proposes K tokens
  Large model verifies all K in one forward pass
  Requires: --draft-model, --draft-config (two checkpoints in memory)

With MTP enabled (v3 addition):
  Large model's own MTP heads propose K = num_future_tokens tokens
  Large model's main head verifies all K in the same forward pass
  Requires: one checkpoint only
```

The same modified-rejection-sampling acceptance rule from v2 §29
applies unchanged — MTP heads are simply a different, cheaper source of
proposal tokens, not a change to how proposals get accepted or rejected.

`mtp: None` on the model config reproduces v1/v2's exact next-token-only
training and inference — MTP heads, when configured, are present but
simply unused (not removed from the checkpoint) at plain
non-speculative inference time.

---

## 42. On-Policy Distillation

v1/v2 have no distillation pipeline — fine-tuning methods (LoRA/QLoRA,
DoRA, DPO) all adapt a model directly on supervised or preference data
the model did not itself generate. On-policy distillation, in the new
`aarambh-ai-distill` crate, is a genuinely different training loop: the
student is trained on *its own* generated rollouts, scored by a teacher,
rather than on independently-generated teacher text.

### 42.1 Why on-policy (not offline) distillation

Plain offline distillation trains a student on text the *teacher*
generated. At the student's own inference time, it will generate text in
its own style and with its own error patterns — text the offline
training data never covered, since the teacher never made those specific
mistakes. This train/inference distribution mismatch means offline
distillation is always training the student on someone else's errors,
not its own. On-policy distillation closes that gap by scoring the
student's actual outputs directly, the same insight already underlying
v1's Online GRPO loop (`SELF_LEARNING.md` §5) — here applied with a
teacher's judgment as the scoring signal instead of a deterministic
verifier's pass/fail.

### 42.2 Pipeline

```rust
// aarambh-ai-distill/src/rollout.rs
pub fn generate_student_rollouts(
    student: &InferenceSession,   // reuses aarambh-ai-inference directly
    prompts: &[String],
    sampling: &SamplingConfig,
) -> Result<Vec<Rollout>> {
    // No gradient tracking here — this is a plain forward-generation
    // pass identical in kind to any other inference call.
}

// aarambh-ai-distill/src/teacher_score.rs
pub trait TeacherScorer {
    /// Scores or corrects a student-generated rollout. Returns a scalar
    /// quality signal (and optionally a corrected/preferred completion,
    /// for a DPO-style variant of the objective).
    fn score(&self, prompt: &str, student_rollout: &str) -> Result<TeacherScore>;
}

pub struct LocalCheckpointTeacher { /* wraps a larger aarambh-ai checkpoint */ }
pub struct ScoredDatasetTeacher { /* looks up pre-scored reference data */ }
// Both implement TeacherScorer — the training loop below is identical
// regardless of which backend is in use.

// aarambh-ai-distill/src/distill_loss.rs
pub fn distill_loss(
    student_logits: &Tensor,       // gradient flows through this
    student_rollout_tokens: &Tensor,
    teacher_score: &TeacherScore,  // gradient does NOT flow through this
) -> Result<Tensor> {
    // Policy-gradient-style objective: increase log-probability of
    // teacher-approved tokens/spans, decrease it for teacher-disapproved
    // ones, weighted by the score's magnitude — structurally the same
    // advantage-weighting math as v1's GRPO (ARCHITECTURE.md §12.4),
    // with the teacher's score standing in for the verifier's reward.
}
```

### 42.3 Training loop shape

```
Repeat for each distillation step:
  1. generate_student_rollouts()  — forward pass only, no gradients
  2. TeacherScorer::score() for each rollout
  3. distill_loss() — gradients computed and applied, student weights
     updated
  4. (if MTP is enabled, §41) MTP auxiliary loss blends into this same
     step's total loss as an additional dense signal, unrelated to but
     computed alongside the teacher-scoring objective
```

Structurally, this alternates inference-mode and training-mode passes in
exactly the same rhythm as v1's Online GRPO loop
(`SELF_LEARNING.md` §5) — the crate is deliberately built to reuse that
existing rhythm rather than invent a new training-loop shape.

---

## 43. Native QAT

v1's quantization (`ARCHITECTURE.md` §16) is entirely post-hoc: train in
full precision end to end, then convert to INT4/GGUF afterward. This
means the model's weights are never trained with any awareness that
they'll eventually be rounded — quantization noise is something the
model only encounters at inference time, on weights that were never
optimised to tolerate it.

### 43.1 Fake quantization

```rust
pub struct FakeQuantize {
    pub bits: QuantBits,       // Int4 | Int8, reuses v1's existing enum
    pub granularity: QuantGranularity, // per-tensor | per-channel, reuses
                                        // v1's existing scale/zero-point
                                        // calculation from ARCHITECTURE.md §16
}

impl FakeQuantize {
    /// Forward: quantize then immediately dequantize, so the numerics
    /// the rest of the forward pass sees match exactly what a truly
    /// quantized model would produce.
    pub fn forward(&self, weight: &Tensor) -> Result<Tensor> {
        let (scale, zero_point) = compute_scale_zero_point(weight, self.granularity)?; // v1's existing fn, unchanged
        let quantized = round_to_grid(weight, scale, zero_point, self.bits)?;
        dequantize(&quantized, scale, zero_point)
    }

    /// Backward: straight-through estimator. The true gradient of the
    /// round() operation is zero almost everywhere, which would kill
    /// learning entirely — STE instead passes the incoming gradient
    /// through unmodified, as if forward() had been the identity
    /// function, clipped to the representable range.
    pub fn backward(&self, grad_output: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let (min_val, max_val) = representable_range(self.bits);
        clip_gradient_outside_range(grad_output, weight, min_val, max_val)
    }
}
```

No new quantization math is introduced — `compute_scale_zero_point()`
and the quantize/dequantize routines are v1's existing functions
(`ARCHITECTURE.md` §16), simply invoked earlier, inside the forward
pass, and made differentiable via the straight-through estimator rather
than applied once as a final, non-differentiable conversion step.

### 43.2 Retrofit recipe

```
1. Start from an existing full-precision checkpoint (any scale, any
   attention/MoE configuration from §38–40)
2. Wrap the configured layers' weights (by default: linear/FFN weights
   only — norms and embeddings stay unwrapped, matching v1's existing
   INT4 scope in §16) in FakeQuantize
3. Continue training for a short, documented number of steps (default:
   ~2% of original pretraining token budget) — a retrofit recipe, not a
   from-scratch requirement, following the same pattern §38 established
4. After training, export via v1's existing GGUF conversion path
   completely unchanged — QAT changes what the weights ARE (now robust
   to quantization noise), not how the export step works
```

### 43.3 Validation

```
Compare, at the same target bit-width:
  (a) post-hoc quantized: full-precision checkpoint → v1's existing
      one-shot INT4 conversion
  (b) QAT: full-precision checkpoint → §43.2's retrofit recipe →
      the SAME one-shot INT4 conversion path

Eval-harness score drop (full-precision baseline → quantized) is
measured for both (a) and (b) on the same holdout. QAT is considered
validated only if (b)'s score drop is measurably smaller than (a)'s —
the actual point of the phase, not assumed from the mechanism alone.
```

`qat: None` on the model config trains exactly as v1 always has — full
precision throughout, with v1's existing post-hoc path remaining
available and unremoved for configs that don't need QAT's extra
retrofit step.

---

## 44. Video Understanding

v2's vision pipeline (`ARCHITECTURE_V2.md` §24–25) is images-only: a
frozen ViT encoder produces patch embeddings, a trainable `Projector` MLP
maps them into the language model's hidden width, and
`interleave_image_tokens()` splices the projected tokens into the
surrounding text token stream at the position of an `<image>` marker.

### 44.1 Frame sampling

```rust
pub struct FrameSampler {
    pub target_frame_count: usize,   // default 8
    pub strategy: SamplingStrategy,
}

pub enum SamplingStrategy {
    /// Frames taken at even intervals across the video's duration.
    Uniform,
    /// Frames concentrated around detected shot/scene boundaries
    /// (simple frame-difference heuristic, not a learned model) —
    /// useful when the video has few, information-dense cuts rather
    /// than continuous smooth motion.
    SceneChangeAware { min_gap_frames: usize },
}
```

Each sampled frame is decoded to a plain image and flows through the
*exact same* frozen ViT encoder v2 already uses — there is no
video-specific encoder architecture. This is a deliberate reuse
decision: the frozen encoder's per-image representations are already
general-purpose visual features; video-specific structure is added
entirely in the fusion step that follows, not in the encoder itself.

### 44.2 Temporal fusion

```rust
pub struct TemporalPositionEmbedding {
    /// One additional embedding vector per sampled frame index,
    /// added to that frame's projected patch tokens before fusion.
    /// Learned (a small embedding table indexed by frame position)
    /// by default; a sinusoidal variant is available as a
    /// parameter-free alternative, following the same learned-vs-
    /// sinusoidal choice v1's RoPE section already documents for
    /// token position.
    embeddings: EmbeddingKind,
}

pub fn interleave_video_tokens(
    text_tokens: &[TokenId],
    frame_token_blocks: &[Vec<Tensor>],   // one Vec<Tensor> per sampled frame
    temporal_embed: &TemporalPositionEmbedding,
    video_marker_position: usize,
) -> Vec<Tensor> {
    // Extends v2's interleave_image_tokens() (ARCHITECTURE_V2.md §24)
    // to handle a SEQUENCE of frame-token blocks rather than a single
    // image's tokens. Each block has the corresponding frame's temporal
    // embedding added before splicing. The single-frame case
    // (frame_token_blocks.len() == 1) must reduce to byte-identical
    // output versus v2's interleave_image_tokens() — enforced by a
    // regression test, not just documented as intent.
    ...
}
```

Without the temporal embedding, a sequence of sampled frames would be
architecturally indistinguishable from one shuffled batch of unrelated
images — the model would have no signal that frame 3 comes after frame 2,
only that some set of images co-occurred in context. This is the one
genuinely new trainable component video adds; everything else (encoder,
base projector shape) is reused unchanged from v2.

### 44.3 Data schema and special tokens

```json
{"video_path": "clips/v001.mp4", "question": "What happens after the ball leaves the ramp?", "answer": "It rolls across the table and falls off the edge.", "num_frames": 8}
```

New reserved tokens `<video>`, `<video_end>`, `<frame_sep>` are allocated
with IDs immediately following v2's existing `<image>`/`<image_end>`
reserved range (`ARCHITECTURE_V2.md` §24), keeping the tokenizer's
reserved-ID block contiguous and documented in one place.

### 44.4 Fine-tuning

Video instruction tuning reuses v2 Phase 20's DoRA-adapted VLM training
path (`vlm_dora.rs`) directly — the loss-masking scheme, the adapter
placement, and the optimiser settings are all unchanged from v2; the
only difference is that a training example now supplies a *sequence* of
frame embeddings (via §44.2's fusion path) rather than one image's
embeddings.

This mirrors, deliberately, how v2's own vision integration was framed
as a reuse of the existing token-stream architecture rather than a
separate model bolted alongside — video extends that same principle one
level further, keeping the frozen-encoder-plus-trainable-projector
economics that make v2's vision approach cheap to train in the first
place.

---

## 45. Document Understanding

Documents (PDFs, scanned pages, multi-column layouts, tables) reuse the
*exact same* frozen ViT encoder as images (§v2) and video (§44) — a
document is treated as a sequence of rendered page-images via
`PageRasterizer`, not as a separately-architected modality with its own
encoder.

### 45.1 Page rasterisation

```rust
pub struct PageRasterizer {
    pub target_dpi: u32,          // default 150 — enough resolution for
                                   // typical body text to remain legible
                                   // through the frozen ViT's patch size
    pub max_pages_per_document: usize, // default 16
}
```

Born-digital PDFs and scanned/photographed pages are both handled the
same way at this layer — both become a sequence of page-images before
anything downstream sees them, so the rest of the pipeline (encoder,
projector, fusion) has no separate code path for "PDF" versus "scanned
image."

### 45.2 Layout-aware projection

```rust
pub struct LayoutAwareProjector {
    /// Extends v2's plain Projector MLP (ARCHITECTURE_V2.md §24) with
    /// 2D positional information per patch: its (row, col) position on
    /// the rendered page, in addition to the 1D sequence position a
    /// plain ViT patch stream already carries implicitly.
    base_projector: Projector,          // v2's original MLP, unchanged
    row_embedding: EmbeddingKind,       // new
    col_embedding: EmbeddingKind,       // new
}

impl LayoutAwareProjector {
    pub fn forward(&self, patches: &Tensor, patch_grid: (usize, usize)) -> Result<Tensor> {
        let projected = self.base_projector.forward(patches)?;
        let (rows, cols) = patch_grid;
        // Add row/col positional information per patch before returning,
        // so patches at the same 1D sequence position but different
        // page locations (e.g. "start of a table cell" vs "start of a
        // paragraph") remain distinguishable downstream.
        add_2d_position(&projected, rows, cols, &self.row_embedding, &self.col_embedding)
    }
}
```

This is a **position-augmentation** change, not an OCR pipeline —
aarambh-ai does not separately parse, segment, or OCR tables/columns
before this step. The model is expected to learn layout structure
(e.g. "text inside a table cell reads differently from a paragraph")
directly from position-augmented patches plus instruction-tuning data,
staying consistent with v2's "vision-language reasoning over raw pixels"
framing rather than adopting a hybrid OCR-plus-LLM design, which would
introduce a separate, non-differentiable parsing stage v2's architecture
was deliberately built to avoid.

### 45.3 Data schema and special tokens

```json
{"document_path": "invoices/inv_042.pdf", "question": "What's the total in the table on page 2?", "answer": "$4,230.00", "pages": [2]}
```

New reserved tokens `<document>`, `<document_end>`, `<page_sep>` follow
the same contiguous-ID-block convention as §44's video tokens.

### 45.4 Fine-tuning

Document instruction tuning reuses the *same* DoRA-adapted VLM training
path as image (v2 Phase 20) and video (§44) tuning once again — by this
point, one training code path in `aarambh-ai-finetune` serves all three
visual modalities, differentiated only by which JSONL schema
(`VqaExample`, `VideoQaExample`, `DocQaExample`) feeds it and which
projector (`Projector` vs `LayoutAwareProjector`) sits between the frozen
encoder and the fusion step.

---

## 46. Long-Horizon Tool-Use Chains

v2 Phase 26 (`ARCHITECTURE_V2.md` §30) shipped a deliberately narrow
scope: single-call, emit-only tool use, constrained by grammar-based
decoding so the model is structurally incapable of emitting invalid
JSON inside a tool-call span. The model produces one typed `ToolCall`;
it does not execute it, and multi-step chains were explicitly deferred
to a later phase.

### 46.1 Chain orchestration

```rust
// aarambh-ai-agent/src/chain.rs
pub struct ToolChain<'a> {
    inference: &'a InferenceSession,   // v2's existing single-call
                                        // decoding path, unmodified
    tools: Vec<ToolDefinition>,        // v2's existing type, unmodified
    max_steps: usize,                  // explicit budget, default 8
}

pub enum ChainStep {
    ToolCall(ToolCall),
    FinalResponse(String),
}

impl<'a> ToolChain<'a> {
    pub fn run(&self, initial_prompt: &str, state: &mut ChainState) -> Result<String> {
        for _ in 0..self.max_steps {
            let step = self.inference.decode_next_step(state.context())?; // v2's path, called repeatedly
            match step {
                ChainStep::FinalResponse(text) => return Ok(text),
                ChainStep::ToolCall(call) => {
                    let result = state.await_caller_supplied_result(&call)?; // caller executes; see boundary note below
                    state.ingest(call, result)?;
                }
            }
        }
        Err(ChainError::MaxStepsExceeded)
    }
}
```

Each step reuses `aarambh-ai-inference`'s existing single-call tool-call
decoding path *completely unmodified* — `ToolChain` is purely an
orchestration loop around a primitive v2 already validated, not a
reimplementation of tool-call decoding.

### 46.2 Typed, multimodal tool results

```rust
pub enum ToolResult {
    Text(String),
    Image(Tensor),      // routes through v2's existing image fusion path
    Video(Vec<Tensor>), // routes through §44's video fusion path
    Document(Vec<Tensor>), // routes through §45's document fusion path
}
```

Building directly on §44–45, a chain step can hand back a screenshot or
a retrieved PDF page and have the *next* step reason over it using the
same multimodal fusion path the rest of the model already uses — not a
text-only summary of it. This is the reason Phase 37 is sequenced after
Phases 35–36 in `ROADMAP_V3.md`: multimodal tool results are only
useful once the fusion machinery to actually consume them exists.

### 46.3 Context management

```rust
pub struct ChainState {
    history: Vec<(ToolCall, ToolResult)>,
    eviction_policy: EvictionPolicy,
}

pub enum EvictionPolicy {
    /// Drop the oldest tool-call/result pairs once context approaches
    /// a configured token budget, keeping the most recent N steps in
    /// full.
    DropOldest { keep_recent_steps: usize },
    /// Summarise dropped steps into a short text digest (via a
    /// dedicated summarisation prompt) rather than discarding them
    /// entirely — a real concern even with §38–39's improved
    /// long-context attention stack, since long-horizon chains can
    /// still exceed any practical ceiling.
    Summarise { keep_recent_steps: usize },
}
```

### 46.4 The boundary stays where v2 drew it

Nothing about this crate changes tool *execution* semantics: `ToolChain`
orchestrates typed requests and ingests typed results supplied by the
caller — it does not sandbox, authorize, or execute tool calls itself.
That responsibility remains exactly where v2 §30 placed it, on the
caller. Multi-step orchestration is new; the emit/execute boundary is
not.

### 46.5 Multi-step SFT

```
Extends v2 Phase 26's tool_sft.rs loss-masking scheme (mask everything
except tool-call spans and the final response) across full multi-turn
transcripts rather than single-call examples — the model is trained on
realistic sequences of "call tool → receive result → decide whether
another call is needed" rather than only ever seeing one call per
example, which is what actually teaches it when to stop.
```

---

## 47. Forgetting Diagnostics

Every self-learning phase since v1 (`SELF_LEARNING.md` §8) has protected
against catastrophic forgetting via gradient orthogonalisation, and v2
extended those protections to vision-grounded sessions
(`SELF_LEARNING_V2.md` §17). Neither version has had a *direct
measurement* of forgetting attached to that defence — its effectiveness
has only ever been inferred indirectly, by watching for score
regressions on the existing eval harness after the fact.

### 47.1 Capability probes

```rust
pub struct CapabilityProbe {
    pub capability: Capability, // Math | Code | Reasoning | Factual |
                                 // Vision | Video | Document | ToolUse
    pub examples: Vec<ProbeExample>, // small, fixed, held-out — built
                                      // from the eval harness's EXISTING
                                      // task subsets (ARCHITECTURE_V2.md
                                      // §17-adjacent eval crate), not new
                                      // benchmark data authored for this
                                      // phase specifically
}
```

### 47.2 Tracking forgetting over a sequence, not just before/after

```rust
pub struct ForgettingCurve {
    capability: Capability,
    /// (checkpoint_or_session_id, score) pairs across an ordered
    /// sequence of checkpoints or online-learning sessions — not just
    /// two points. This is what makes gradually-developing forgetting
    /// visible, as distinct from forgetting caused by one discrete
    /// event.
    points: Vec<(String, f64)>,
}

pub fn forgetting_delta(curve: &ForgettingCurve, baseline_id: &str, current_id: &str) -> Option<ForgettingDelta> {
    let before = curve.score_at(baseline_id)?;
    let after = curve.score_at(current_id)?;
    let delta = after - before;
    Some(ForgettingDelta {
        delta,
        // Below this magnitude, treated as noise rather than real
        // forgetting — default threshold 0.02 (2 percentage points),
        // documented and configurable, not a hidden magic number.
        significant: delta.abs() >= SIGNIFICANCE_THRESHOLD,
    })
}
```

### 47.3 Measurement-only, by design

This diagnostic layer explicitly does not alter training gradients or
introduce a new defence mechanism of its own — the existing gradient-
orthogonalisation defence (`SELF_LEARNING.md` §8) remains the actual
protection. Phase 38's job is entirely to measure whether that existing
defence is holding, not to replace or supplement it with a second
mechanism. This separation is deliberate: conflating "measuring
forgetting" with "preventing forgetting" would make it impossible to
tell whether an observed improvement came from a better defence or from
a change in how forgetting was being measured.

### 47.4 Shared export schema — tied to Manas

```json
{
  "capability_or_concept": "string identifier",
  "baseline_checkpoint_or_session": "string identifier",
  "current_checkpoint_or_session": "string identifier",
  "score_before": 0.0,
  "score_after": 0.0,
  "delta": 0.0,
  "significant": true
}
```

The diagnostics export in this documented schema specifically so that
aarambh-ai's capability-level forgetting curves and Manas's own
concept-level forgetting curves (a separate, from-scratch associative-
memory system with its own anti-forgetting design) can sit in the same
shape of record — full detail on Manas's side of this connection is in
`SELF_LEARNING_V3.md` §28. This is a documentation-level convention, not
a runtime dependency between the two codebases: neither project
automatically consumes the other's curves, but both are readable in the
same vocabulary, which is the actual value on offer.

---

## 48. Max Thinking Mode

v1 established four thinking modes: None (0 tokens) → Low (≤256) →
Medium (≤1,024) → High (≤4,096), each a hard budget on how many
`<think>`-block tokens the model may spend before being force-closed
into its final answer (`ARCHITECTURE.md` §7.2–7.4). Max is the fifth
mode, sitting above High at a 16,384-token budget — the next step in the
same roughly-4x progression the existing four modes already follow
(0 → 256 → 1,024 → 4,096 → 16,384).

### 48.1 No new mechanism, only a new budget value

```rust
pub enum ThinkingMode {
    None,
    Low,     // budget() = 256
    Medium,  // budget() = 1_024
    High,    // budget() = 4_096
    Max,     // budget() = 16_384   ← v3 addition
}

impl ThinkingMode {
    pub fn budget(&self) -> usize {
        match self {
            ThinkingMode::None => 0,
            ThinkingMode::Low => 256,
            ThinkingMode::Medium => 1_024,
            ThinkingMode::High => 4_096,
            ThinkingMode::Max => 16_384,
        }
    }
}
```

`ThinkingController` (`ARCHITECTURE.md` §7.4) already handles arbitrary
budgets generically — `take_forced_token()` forces `<think>` open once
when `mode != ThinkingMode::None`, and `on_token()` counts thinking
tokens, force-closing with `</think>` once `tokens_used >= self.mode.budget()`.
Adding Max is a one-variant, one-match-arm change: the force-close
logic, the `raw_text`/`thinking_text`/`answer_text` separation, and the
CLI's collapsed-thinking rendering all continue to work completely
unmodified, exactly as they did when High was the largest mode —
verified by a regression test asserting `None`/`Low`/`Medium`/`High`
behaviour is byte-for-byte unchanged after Max is added.

### 48.2 Why it still needs its own training pass

A larger budget alone does not make a model reason better inside it —
Stage 2's GRPO training (`ARCHITECTURE.md` §7.5) is what teaches the
model to allocate thinking length to match a problem's actual
difficulty, and that allocation policy has, up to v2, only ever been
trained against problems where High's 4,096-token ceiling was
sufficient to reach a correct answer. Phase 39 re-runs Stage 2 GRPO
specifically including Max-budget rollouts, on a held-out set of
problems selected because they were *not* reliably solvable within
High's budget:

```
Candidate "High-insufficient" problem sources for the re-run's training
mix:
  - Multi-stage proofs requiring several independent lemmas
  - Dense multi-file code changes where the reasoning must track state
    across more context than a single High-budget block can hold
  - Long-horizon tool-chain planning (§46) — reasoning through an entire
    multi-step plan before the first tool call is even made
```

Without training signal from problems that genuinely need the extra
room, Max mode would simply spend more tokens for the same outcome as
High — which the existing reward shaping (concise-and-correct → high
reward; excessive, unproductive thinking → penalised, unchanged from
`ARCHITECTURE.md` §7.5) is specifically designed to catch and discourage
even at Max's larger ceiling.

### 48.3 Sampling defaults

Extending v1's existing per-mode table (`ARCHITECTURE.md` §8.2):

| Mode | Temperature | Top-p |
|---|---|---|
| None | 0.7 | 0.9 |
| Low | 0.75 | 0.92 |
| Medium | 0.8 | 0.95 |
| High | 0.8 | 0.95 |
| **Max (v3)** | **0.85** | **0.97** |

Max mode is the most exploratory of the five: Max-mode tasks are
exactly the ones where premature convergence on a wrong early step is
most costly, given how much subsequent reasoning would then be built on
that mistake, so sampling stays looser for longer than High's already-
exploratory defaults.

### 48.4 Validation

Phase 39's milestone is a documented accuracy improvement over High
mode, specifically on the held-out High-insufficient problem set, via
the eval harness's new "hard problems" task — the same "does the trade
pay off" discipline v2 Phase 22 applied to MoE-versus-dense and v3
§40.5 applied to fine-grained-versus-coarse routing. A larger budget
that doesn't measurably improve accuracy on the problems it was built
for is not considered a validated feature, regardless of how reasonable
the mechanism sounds in isolation.
---

## 49. Updated Dependency Layers

```
Layer 0  aarambh-ai-core
Layer 1  aarambh-ai-tokenizer   aarambh-ai-data
Layer 2  aarambh-ai-nn          aarambh-ai-kernel
Layer 3  aarambh-ai-model       aarambh-ai-weights    aarambh-ai-quant     aarambh-ai-vision
Layer 4  aarambh-ai-train       aarambh-ai-finetune
Layer 5  aarambh-ai-inference   aarambh-ai-safety     aarambh-ai-selflearn
         aarambh-ai-eval        aarambh-ai-distill    aarambh-ai-agent
Layer 6  aarambh-ai-serve       aarambh-ai (binary)
```

`aarambh-ai-distill` sits at Layer 5 alongside `aarambh-ai-eval`: it needs
a fully-assembled, invokable model (both student and teacher) to generate
and score rollouts, the same requirement that placed eval at Layer 5 in
v2. `aarambh-ai-agent` sits at Layer 5 as well — it orchestrates calls
into the Layer 5 inference/safety stack, similar in kind to how
`aarambh-ai-serve` (Layer 6) wraps that same stack with an HTTP transport,
except agent chains are a orchestration layer rather than a new entry
point, so they sit one layer lower, alongside the components they call.

The same rule as v1/v2 applies unchanged: every crate may only depend on
crates in the same or lower layer, enforced by `Cargo.toml`.

### New Dependency Policy Entries

| Dependency | Allowed crates | Reason |
|---|---|---|
| Video container decode crate (pure-Rust or permissively-licensed) | `aarambh-ai-vision` | Frame extraction only, no network calls |
| PDF/document rasterisation crate | `aarambh-ai-vision` | Page-to-image rendering only, no network calls |

**Still forbidden everywhere, unchanged from v1/v2:** PyTorch bindings
(`tch-rs`), ONNX Runtime (`ort`), Python FFI, `llama.cpp` as a backend.
Video and document rasterisation crates must not depend on Python-based
video/PDF ML tooling — pure-Rust or permissively-licensed system-library
bindings only, consistent with the existing `image`-crate precedent from
v2 §32.

**Version policy:** unchanged from v1/v2 — pin major versions, test the
whole workspace on any `candle-core` upgrade.

---

## 50. Updated Memory & Compute Estimates

### Attention Stack (relative to v1/v2's full-attention baseline)

| Component | KV-cache / state memory at long context | Notes |
|---|---|---|
| Full attention (v1/v2 baseline) | Grows linearly with generated tokens | Unchanged, still available as fallback |
| Gated DeltaNet layer | Constant, independent of generated tokens | Fixed-size recurrent state (§38) |
| DSA sparse layer | Grows linearly, but with a smaller per-token constant (only selected blocks materialise) | Full attention's asymptotic behaviour, smaller multiplier |

### Fine-Grained MoE (relative to v2's dense MoE, `num_experts=8, top_k=2` example)

| | v2 Dense MoE | v3 Fine-Grained MoE (`fine_grained_factor=4, num_shared_experts=1`) |
|---|---|---|
| Distinct experts | 8 | 32 routed + 1 shared |
| Active params/token | ≈ dense-equivalent (top_k=2 of 8) | ≈ dense-equivalent (top_k of 32) + shared expert (always active) |
| Total params | Grows with `num_experts` | Grows further with `num_experts × fine_grained_factor`, active compute still close to dense-equivalent |

The shared expert's fixed cost is small relative to the routed pool and
is accounted for separately from the load-balancing loss (§40).

### MTP Heads

| Component | Params | Notes |
|---|---|---|
| Each MTP head | One additional lightweight transformer block + LM head per `num_future_tokens` offset | Training-time only by default; reusable as speculative-decode draft (§41) |

### QAT

QAT changes *when* quantization happens, not the final quantized memory
footprint — INT4/INT8 memory numbers are unchanged from v1's existing
post-hoc quantization table (`ARCHITECTURE.md` §16). The cost QAT adds is
training-time only: `FakeQuantize`'s forward/backward pass adds a modest
compute overhead during the short continued-training recipe, not a
standing memory cost.

### Video/Document Vision Addition (on top of v2's per-image table, §33)

| Component | Params | Notes |
|---|---|---|
| Frozen ViT encoder | Unchanged from v2 (~86M, ~344 MB) | Reused as-is across image/video/document |
| `TemporalPositionEmbedding` | Small (learned or sinusoidal) | Added per frame, negligible relative to encoder |
| `LayoutAwareProjector` | Slightly larger than v2's plain `Projector` MLP | 2D positional augmentation adds a modest parameter increase over v2's 1D projector |

Video specifically multiplies the *activation* memory cost by the number
of sampled frames (each frame runs the same frozen encoder forward pass);
this is a compute/activation cost, not a parameter-count increase, since
the encoder itself is shared and frozen across frames.

### Max Thinking Mode (relative to v1's existing budget table)

| Mode | Token Budget | Notes |
|---|---|---|
| None (v1) | 0 | Unchanged |
| Low (v1) | ≤256 | Unchanged |
| Medium (v1) | ≤1,024 | Unchanged |
| High (v1) | ≤4,096 | Unchanged |
| Max (v3, §48) | ≤16,384 | New; same `ThinkingController` mechanism, no new memory-management path |

A larger thinking budget increases *generation-time* token count for
sessions that use it, not parameter or checkpoint memory — Max mode adds
no new weights of its own (unlike MTP heads, §41, or MoE experts, §40).
The practical cost is more KV-cache/recurrent-state turnover during a
long `<think>` block, which is exactly where §38's Gated DeltaNet
constant-memory state and §39's DSA sparse layers pay off most: Max-mode
sessions are the long-generation case those two mechanisms were built
for.

---

## 51. Updated Hardware Strategy

### Your Local Machine (i3-1115G4, 8 GB RAM, Pop OS)

Everything from v1's and v2's i3 capability lists still applies unchanged.
v3.0 adds:

- Tool-chain orchestration (`aarambh-ai-agent`) for Tiny/Small checkpoints
  — orchestration overhead is lightweight; the underlying inference calls
  follow the same i3-capability rules v2 already established for
  single-call tool use
- QAT-trained checkpoint inference (once trained on Kaggle) runs on i3
  exactly as v1's post-hoc-quantized checkpoints already do
- Forgetting-diagnostic capability probes on Tiny/Small (small, fixed
  held-out sets — same i3-capability class as v2's eval-harness subsets)
- Max-mode *inference* on an already-trained Tiny/Small checkpoint —
  `ThinkingController`'s force-close mechanism is the same lightweight
  CPU-side logic at any budget value; only the Max-mode GRPO *retraining*
  step below needs Kaggle

**Explicitly NOT i3-capable in v3.0:**
- Gated DeltaNet/DSA retrofit training (continued pretraining, Kaggle-scoped)
- Fine-grained MoE training (dispatch overhead assumes GPU-scale batches,
  same reasoning as v2 §34)
- MTP training (auxiliary heads add real per-step training compute)
- On-policy distillation (requires running both student rollout
  generation and teacher scoring, Kaggle-scoped)
- Video/document encoder training runs (frozen-encoder forward passes at
  multi-frame/multi-page scale add real per-turn cost, same class of
  reasoning as v2's vision-self-learning Kaggle-only gate,
  `SELF_LEARNING_V2.md` §19)
- Max-mode GRPO retraining (Stage 2 re-run generating G=8 rollouts per
  problem at up to 16,384 thinking tokens each is Kaggle-scoped, same
  class of cost as v1's original Stage 2 GRPO run); plain Max-mode
  *inference* on an already-trained Tiny/Small checkpoint remains
  i3-capable — only the retraining step needs Kaggle

### Kaggle GPU (unchanged tiers, new workloads)

| Scale/Workload | GPU | Notes |
|---|---|---|
| Gated DeltaNet / DSA retrofit | T4/P100 | Continued pretraining from an existing checkpoint |
| Fine-grained MoE training | P100/A100 recommended | Larger expert pool benefits from more VRAM headroom, same reasoning as v2 §34 |
| MTP training | T4/P100 | Auxiliary heads add modest compute over base pretraining |
| On-policy distillation | T4/P100 | Needs both student and teacher loaded (or teacher as a scored dataset) |
| Native QAT | T4 | Short continued-training recipe, same cost class as other retrofits |
| Video/document training | T4/P100 | Frozen encoder reused, multi-frame/multi-page batches add activation memory |
| Long-horizon tool chains (SFT) | T4/P100 | Multi-turn transcripts, same cost class as v2 Phase 26 |
| Forgetting diagnostics | T4 | Lightweight probes, runs alongside existing training/self-learning loops |
| Max-mode Stage 2 GRPO retraining | T4 | G=8 rollouts per problem at up to 16,384 thinking tokens; same cost class as v1's original Stage 2 GRPO run, just longer rollouts |

---

## 52. What's Explicitly Out of Scope

Carried over and extended from `ROADMAP_V3.md` for architectural
completeness:

- Releasing any pretrained checkpoint, adapter, or GGUF file
- Sparse/grouped MoE dispatch (still dense-masked-matmul, carried forward
  from v2 §35 a second time)
- Multi-node distributed training (still 2-GPU single-node only)
- Public/hosted deployment of the inference server (still local-only)
- Audio modality (video understanding is visual frames only, no audio
  track processing)
- Tool execution/sandboxing (Phase 37 orchestrates and ingests results
  only; it does not execute or authorize tool calls)
- Synthetic API-generated training data (all v3.0 datasets remain free
  and public)

These are natural v4 candidates once budget and a released model exist.
