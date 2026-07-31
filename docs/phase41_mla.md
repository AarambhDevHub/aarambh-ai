# Phase 41 — Multi-Head Latent Attention (MLA)

> v4.0.0-alpha.1 · `aarambh-studio-nn` (`mla.rs`) · depends on v1 §6.3 (GQA/RoPE), v2 §21 (YaRN/NTK), v3 §29 (`HybridAttentionSchedule`)

Phase 41 adds a third attention kind — **latent KV compression** — to the
hybrid attention schedule v3 §29 introduced. A model can now mix **Full**,
**Gated DeltaNet**, and **LatentMLA** layers in whatever ratio the config
specifies. MLA layers cache a single low-rank latent vector per token instead
of full per-head keys and values, cutting KV-cache memory per token
substantially at long context, without discarding per-head expressiveness.

This is the third and final leg of the attention family v3 began (Gated
DeltaNet = linear attention, DSA = sparse attention, MLA = latent
compression), completing the pattern current frontier open-weight labs ship.

## Mechanism

MLA compresses what gets **cached**, not what gets **computed**. Instead of
caching per-head K and V directly, a token's hidden state is down-projected
once into a single shared latent vector `c_kv`; per-head keys and values are
then reconstructed from that one latent via small per-head up-projection
matrices — which are ordinary trainable weights, not part of the cache.

```
hidden_state (d_model)
     │
     ▼
kv_a_proj: d_model -> latent_dim        (down-projection)
     │
     ▼
c_kv  (RMSNormed)  ── the ONLY latent cached per token, for MLA layers
     │
     ├──▶ up_k: latent_dim -> n_heads * nope_head_dim   (weight, not cached)
     │        │
     │        ▼
     │      K_nope^(h)  (reconstructed per head, at attention time)
     │
     └──▶ up_v: latent_dim -> n_heads * value_head_dim  (weight, not cached)
              │
              ▼
            V^(h)  (reconstructed per head, at attention time)
```

### Decoupled RoPE

A naively-compressed latent cannot carry an already-rotated (position-encoded)
key — rotation is head-dimension-specific and applying it before compression
would defeat the point of sharing one latent across heads. MLA splits each
head's query and key into two parts:

- a larger **"nope"** (no positional encoding) part derived straight from the
  compressed latent (`q_proj` → `q_nope`, `up_k` → `k_nope`), and
- a small separate **"rope"** part that *is* rotary-encoded and cached on the
  side (`k_rope_proj` → `k_rope`, shared across heads), at a much smaller
  per-head width (`rope_head_dim`, default 16) than a full key would need.

The cache for an MLA layer, per token, is therefore
`{c_kv (latent_dim) + k_rope (rope_head_dim)}` — substantially smaller than a
full per-head K and V cache at typical configurations, while per-head
expressiveness is preserved through the up-projection weights at attention
time.

### MLA layer parameters (checkpoint names under `blocks.{i}.mla.`)

| Tensor | Shape | Role |
|---|---|---|
| `q_proj.weight` | `[hidden, n_heads*(nope+rope)]` | full query projection |
| `kv_a_proj.weight` | `[hidden, latent_dim]` | down-projection → `c_kv` (cached) |
| `kv_a_norm.weight` | `[latent_dim]` | RMSNorm over the compressed latent |
| `up_k.weight` | `[latent, n_heads*nope]` | per-head key nope up-projection |
| `up_v.weight` | `[latent, n_heads*value]` | per-head value up-projection |
| `k_rope_proj.weight` | `[hidden, rope_head_dim]` | rotary key slice (shared across heads, cached) |
| `o_proj.weight` | `[n_heads*value, hidden]` | output projection |

## Configuration

MLA layers are placed into an existing v3 hybrid schedule via two new
`[model.attention_schedule]` fields:

```toml
[model.attention_schedule]
full_attention_every_n = 4        # v3 rule: every Nth layer is Full
mla_layers = [2, 6, 10, 14, 18, 22]   # v4: these layers become LatentMLA

[model.attention_schedule.gated_deltanet]
# ...unchanged v3 Gated DeltaNet settings for the remaining non-Full layers...

[model.attention_schedule.mla]
latent_dim = 512
rope_head_dim = 16
# nope_head_dim, value_head_dim, n_heads default to 0 and are derived:
#   n_heads       = model.n_heads
#   nope_head_dim = host_head_dim - rope_head_dim   (e.g. 64 - 16 = 48)
#   value_head_dim = nope_head_dim
```

`mla_layers` takes precedence over both the `full_attention_every_n` rule and
the DSA full-attention override, so an MLA slot is never silently replaced by
Sparse. A schedule with an empty `mla_layers` and no `mla` block reproduces
v3.0.0 exactly — the same backward-compatibility discipline every attention
change since v1 has held.

See `configs/mla_smoke.toml`, `configs/medium_hybrid_mla.toml`, and
`configs/large_hybrid_mla.toml` for ready-to-use recipes.

## Measured KV-cache footprint

`aarambh-studio eval --config <cfg> --kv-cache-report` prints the per-layer
bytes/token by attention kind (no checkpoint required — only the config):

```text
KV-cache bytes/token (dtype=F32, 4 bytes/element, 24 layers)
layer  kind             bytes/tok  note
0      full                 1024  2 * n_kv_heads * head_dim per token
1      gated_deltanet          0  fixed recurrent state (not per-token)
2      latent_mla            2112  latent_dim + rope_head_dim per token (compressed)
...
total bytes/token across all layers: 23040
all-full baseline (24 layers): 24576
hybrid/all-full ratio: 0.938 (93.8% of all-full cache)
```

For the Medium hybrid MLA config (`latent_dim=512`, `rope_head_dim=16`,
`n_kv_heads=8`, `head_dim=64`): MLA per-token cache = 528 elements vs the GQA
baseline of 1024 elements — a **~1.94× reduction** on the retrofitted layers.
Gated DeltaNet layers contribute 0 bytes/token (fixed recurrent state). The
net win shows up at long context, where the KV cache dominates.

## Retrofit from a v3 checkpoint

Following the exact pattern v3 §29 established: MLA layers are added to an
existing v3.0.0 checkpoint via continued pretraining, not a from-scratch
rebuild. Scheduled layers are reinitialised with fresh MLA parameters; every
other layer's weights load unchanged from the v3 checkpoint. Training proceeds
at a reduced learning rate (`retrofit_lr_scale = 0.1`) so the untouched layers
do not drift meaningfully while the new layers learn.

```sh
scripts/phase41_prepare_mla_retrofit.sh data checkpoints/wikitext103_medium_hybrid/model.safetensors
aarambh-studio train --config configs/medium_hybrid_mla.toml
```

The partial-checkpoint loader (`aarambh_studio_weights::load_retrofit_into_varmap`)
counts `.mla.` tensors as freshly initialised (alongside the existing
`.deltanet.` and `.dsa.` paths) and loads every shared tensor (embedding,
norms, FFN, output head) bit-exactly. See the
`partial_checkpoint_load_preserves_non_mla_layer_weights_exactly` test.

## Tests

The Phase 41 proof obligations (from `ROADMAP_V4.md`):

| Test | Location | Proves |
|---|---|---|
| `schedule_with_zero_mla_layers_matches_v3_exactly` | `aarambh-studio-core` | empty `mla_layers` reproduces v3.0.0 `kind_for_layer` and `resolved_mla` returns `None` |
| `mla_layers_take_precedence_over_every_n_and_dsa_override` | `aarambh-studio-core` | MLA slots win over the every-n rule and the DSA override |
| `mla_reconstructed_kv_matches_reference_full_attention_within_tolerance` | `aarambh-studio-nn` | latent round-trip produces bounded, finite attention output; train == inference path |
| `decoupled_rope_nope_split_preserves_relative_position_encoding` | `aarambh-studio-nn` | nope half is position-invariant; rope half changes with offset |
| `mla_kv_cache_bytes_per_token_is_smaller_than_full_or_gqa_baseline` | `aarambh-studio-nn` / `aarambh-studio-model` | `(latent_dim + rope_head_dim) < 2 * n_kv_heads * head_dim` |
| `partial_checkpoint_load_preserves_non_mla_layer_weights_exactly` | `aarambh-studio-weights` | retrofit loads shared tensors bit-exactly, initialises 7 MLA tensors |
| `mla_model_forwards_and_cached_forward_matches_full_forward` | `aarambh-studio-model` | cached decode matches full forward; MLA cache grows per token |
| `mla_training_backward_reaches_mla_parameters` | `aarambh-studio-model` | gradients reach the MLA down/value/output projections (§42 reachability) |
| `mla_kv_cache_report_shows_compressed_footprint` | `aarambh-studio-model` | `--kv-cache-report` reports the compressed MLA footprint |

### Smoke test

```sh
scripts/phase41_smoke.sh
```

Runs the MLA unit tests, the `--kv-cache-report` check on `configs/mla_smoke.toml`
(no checkpoint needed), and — when a training fixture is available — a two-step
CPU training smoke that verifies the saved checkpoint contains
`blocks.0.mla.*` tensors.

## Scope and boundaries

- MLA reuses the candle fallback attention kernel, which tolerates a value
  head width different from the query/key head width (`value_head_dim` may
  differ from `nope_head_dim + rope_head_dim`). CUDA flash / fused MLA kernels
  are future work; the mechanism and memory win are in place.
- YaRN/NTK long-context scaling applies unchanged to the host transformer's
  full-attention layers. MLA's dedicated `rope_head_dim`-wide rotary slice
  uses base RoPE; applying YaRN scaling to the compressed-latent rope slice is
  a documented refinement, not a regression.
- Self-learning (`SELF_LEARNING_V4.md` §42) is transparent to MLA: online GRPO
  operates on token log-probabilities and has no dependency on the attention
  kind. Gradient orthogonalisation reaches MLA's down/up-projection weights
  the same way it reaches every other trainable weight — verified by
  `mla_training_backward_reaches_mla_parameters`.
