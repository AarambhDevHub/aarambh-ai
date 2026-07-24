# SELF_LEARNING_V3.md — aarambh-ai v3.0

> Companion to `SELF_LEARNING.md` and `SELF_LEARNING_V2.md`. This document
> covers **only what v3.0 adds** — forgetting diagnostics, and how the
> rest of v3.0's architecture (hybrid attention, fine-grained MoE, MTP,
> distillation, video/document modalities, tool-use chains) interacts
> with the self-learning loop. Sections continue numbering from v2's
> Section 24. Everything in `SELF_LEARNING.md` (Online GRPO, Self-Critique,
> Experience Replay, CPU vs GPU mode, catastrophic forgetting protection)
> and `SELF_LEARNING_V2.md` (vision-grounded verification, extended replay
> schema, Kaggle-only vision gating) is unchanged and continues to work
> exactly as documented for text-only and image-only sessions.

---

## Table of Contents

25. [Why This Is a Separate Document](#25-why-this-is-a-separate-document)
26. [What Changes, What Doesn't](#26-what-changes-what-doesnt)
27. [Forgetting Diagnostics in the Self-Learning Loop](#27-forgetting-diagnostics-in-the-self-learning-loop)
28. [The Manas Connection](#28-the-manas-connection)
29. [Extended Replay Buffer Schema (v3)](#29-extended-replay-buffer-schema-v3)
30. [Self-Learning on the New Attention Stack](#30-self-learning-on-the-new-attention-stack)
31. [Self-Learning with Fine-Grained MoE](#31-self-learning-with-fine-grained-moe)
32. [Video and Document Grounded Verification](#32-video-and-document-grounded-verification)
33. [Self-Learning Inside Long-Horizon Tool Chains](#33-self-learning-inside-long-horizon-tool-chains)
34. [Hardware Gating (v3 Additions)](#34-hardware-gating-v3-additions)
35. [Full Loop Flow (v3, All Modalities)](#35-full-loop-flow-v3-all-modalities)
36. [CLI Commands (v3)](#36-cli-commands-v3)
37. [Crate Structure Additions](#37-crate-structure-additions)
38. [What to Expect (v3)](#38-what-to-expect-v3)
39. [Known Limitations (v3-Specific)](#39-known-limitations-v3-specific)

---

## 25. Why This Is a Separate Document

v3.0's architecture changes (`ARCHITECTURE_V3.md` §38–48) touch enough of
the model that self-learning needs its own design pass again, for
reasons distinct from why v2 needed one:

- v1's catastrophic-forgetting safeguards (`SELF_LEARNING.md` §8) use a
  frozen reference/base model, low-rank adapters, KL regularization, small
  learning rates, and diverse replay. They have never had a persistent
  direct *measurement* attached to them.
  Once the model has hybrid attention, MoE routing, MTP heads, and three
  visual modalities, "did we forget something" stops being a question
  you can answer by eyeballing eval-score deltas after the fact — you
  need the forgetting diagnostics from `ARCHITECTURE_V3.md` §47 wired
  directly into the loop that's doing the updating.
- The **Manas** project (a separate associative-memory system, see
  Darshan's Manas v3 design docs) is independently building its own
  anti-forgetting tracking for a very different architecture — an
  associative-memory network rather than a transformer. The two projects
  converging on a *shared* forgetting-measurement schema is valuable
  specifically because it lets lessons transfer between them without
  each one re-deriving what "forgetting" even means numerically.
- Fine-grained MoE routing (§40 in ARCHITECTURE_V3.md) introduces a new
  forgetting failure mode that v1/v2 never had: an online update can
  silently retrain a routing decision as easily as it retrains a weight
  — a token that used to route to Expert 12 might start routing to
  Expert 3 after a self-learning update. End-to-end capability probes and
  routing signatures are both needed to make that change visible.

## 26. What Changes, What Doesn't

**Unchanged, and verified unchanged by regression tests:**
- Text-only self-learning on i3 — same CPU-safe deferred-gradient mode,
  same replay buffer behaviour, same critique prompt, same eviction
  policy (`SELF_LEARNING.md` §6, §8), byte-for-byte.
- Image-grounded self-learning on Kaggle — same vision-grounded verifier,
  same `image_ref` replay schema, same Kaggle-only hardware gate
  (`SELF_LEARNING_V2.md` §17, §19), byte-for-byte.
- Existing `replay_buffer.jsonl` and `replay_buffer_v2.jsonl` files load
  exactly as before. No migration step is required for text-only or
  image-only users moving to v3.0.

**New, additive only:**
- Opt-in forgetting probes run after each committed online-GRPO update,
  deferred-gradient flush, or replay update (§27), producing a signal the
  loop can log and, in a later phase, act on. v3.0 ships measurement, not
  an automatic corrective action.
- An optional `video_ref` and `document_ref` field on replay entries
  (§29), following the exact caching pattern v2 established for
  `image_ref`.
- MoE-aware replay sampling diagnostics: tracking which experts activate
  for replayed entries, to detect routing drift specifically (§31).
- A documented export schema so forgetting curves can be consumed outside
  aarambh-ai entirely — specifically, by Manas (§28).

## 27. Forgetting Diagnostics in the Self-Learning Loop

`ARCHITECTURE_V3.md` §47 defines `CapabilityProbe`, `ForgettingCurve`, and
`forgetting_delta()` at the `aarambh-ai-eval` crate level. This section
covers how they plug into the *online* self-learning loop specifically,
where the stakes are different from a one-off training run: self-learning
runs continuously, on user-driven data the model has no control over the
distribution of, which is exactly the setting catastrophic forgetting is
most likely to bite.

**Where it hooks in:**

```
Online GRPO update batch completes (SELF_LEARNING.md §5)
        │
        ▼
Run each CapabilityProbe (math, code, reasoning, factual, vision,
video, document, tool-use) — small, fixed, held-out, cheap
        │
        ▼
Compute forgetting_delta() against the pre-session baseline checkpoint
        │
        ▼
Log to ForgettingCurve; if any capability's delta crosses the
significance threshold in the negative direction, flag it in the
session's summary output (visible to you, not silently swallowed)
```

This does **not** change update math or the safeguards in
`SELF_LEARNING.md` §8. Phase 38 observes an in-memory merged adapter view
after a committed update, so deferred CPU gradients are measured at flush
time rather than while still pending. The probes never call backward or an
optimizer.

**Cost.** Capability probes are deliberately small and fixed — the same
class of cost as v1's existing self-critique overhead, not a new
Kaggle-only requirement. They run on i3 for text/tool-use capabilities;
vision/video/document capability probes follow the same Kaggle-only gate
`SELF_LEARNING_V2.md` §19 already established for anything touching the
frozen visual encoder.

## 28. The Manas Connection

Manas is a separate associative-memory project, not a transformer. Its
memory, provenance, and forgetting mechanisms are independent from
aarambh-ai, and this document does not attempt to unify the two systems
technically.

What *is* shared is the **measurement schema**. `forgetting_delta()`'s
export format (`ARCHITECTURE_V3.md` §47) is documented so that a
capability-level forgetting curve computed by aarambh-ai and a
concept-level forgetting curve computed by Manas can sit in the same
shape of record:

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

Neither project consumes the other's curves automatically. Aarambh has no
runtime dependency, path discovery, or write access to `../manas`; the
versioned JSON Schema is an optional operator-controlled bridge. The value
is that when you are reasoning
about *either* project's forgetting behaviour, you are reasoning in the
same vocabulary, and any lessons learned tuning aarambh-ai's significance
threshold or probe design transfer directly to thinking about Manas's
equivalent, and vice versa.

## 29. Extended Replay Buffer Schema (v3)

v2's replay entries (`SELF_LEARNING_V2.md` §16) added an optional
`image_ref`:

```json
{"prompt": "How many cats are in this image?", "response": "There are 3 cats.", "score": 0.95, "timestamp": 1719000000, "topic": "vision", "image_ref": "vision_cache/img_a1b2c3.safetensors"}
```

v3.0 adds two more optional fields, following exactly the same caching
principle — a reference to *cached projected tokens*, not the raw
video/document file, since replay resamples entries repeatedly and
re-running the frozen encoder/projector path on every replay batch would
be wasted compute for components that don't change during a self-learning
session:

```json
{"prompt": "What happens after the ball leaves the ramp?", "response": "It rolls across the table and falls off the edge.", "score": 0.88, "timestamp": 1719000500, "topic": "video", "video_ref": "vision_cache/vid_f8e7d6.safetensors"}
```

```json
{"prompt": "What's the total in the invoice table on page 2?", "response": "$4,230.00", "score": 0.91, "timestamp": 1719000900, "topic": "document", "document_ref": "vision_cache/doc_c3b2a1.safetensors"}
```

**Schema versioning:** buffer files with any video or document entries
use a `replay_buffer_v3.jsonl` name. Existing `replay_buffer.jsonl` (v1)
and `replay_buffer_v2.jsonl` (v2, image-only) files load unchanged, with
`video_ref` and `document_ref` defaulting to `None` for every entry —
enforced by regression tests exactly mirroring v2's
`v1_replay_buffer_loads_with_image_ref_defaulting_to_none` pattern.

**Eviction and sampling** (`SELF_LEARNING.md` §6) are unchanged again —
score²-weighted sampling and the diversity constraint apply identically,
with `"video"`, `"document"`, and `"tool-use"` added as new topics
alongside v1's `math, code, reasoning, factual, creative, general` and
v2's `vision`.

## 30. Self-Learning on the New Attention Stack

Hybrid Gated DeltaNet/DSA attention (`ARCHITECTURE_V3.md` §38–39) changes
how state is carried during generation, which matters for self-learning
specifically because online sessions run many turns back-to-back:

- Gated DeltaNet layers carry a fixed-size recurrent state rather than a
  growing KV cache. This is strictly *better* for long self-learning
  sessions memory-wise — a session that runs for hundreds of turns no
  longer accumulates unbounded KV-cache memory for those layers the way
  v1/v2's all-full-attention design would.
- The online-GRPO update step (`SELF_LEARNING.md` §5) treats Gated
  DeltaNet and DSA layers exactly as it treats any other differentiable
  layer — gradients flow through the chunk-parallel training-time form
  (`ARCHITECTURE_V3.md` §38) during the update, and generation reverts to
  the sequential recurrent form afterward. No special-casing is needed in
  the self-learning loop itself; this is handled entirely at the
  `aarambh-ai-nn` layer.
- The self-learning safeguards remain adapter-scoped and therefore do not
  need a separate Gated DeltaNet/DSA branch. Phase 38 observes the resulting
  end-to-end model exactly as it observes full-attention models.

## 31. Self-Learning with Fine-Grained MoE

Fine-grained MoE (`ARCHITECTURE_V3.md` §40) introduces a forgetting
failure mode v1/v2 never had: **routing drift**. Because routing
decisions are themselves learned, an online update can shift *which*
experts a given kind of input activates. A token that reliably
routed to a "math" specialist expert before a self-learning session might
route somewhere else afterward — the weights for that expert may be
completely intact and still "know" the material, but the router no longer
sends that kind of input there, which looks exactly like forgetting from
the outside.

**Routing-drift diagnostic** (extends §27's forgetting diagnostics):

```
For each CapabilityProbe example, record which expert(s) activated
before the self-learning session (baseline) and after (current)
        │
        ▼
routing_drift_rate = fraction of probe examples whose top-activated
expert(s) changed between baseline and current
        │
        ▼
Logged alongside score-based forgetting_delta(); a high routing-drift
rate with a low score-based delta is itself a signal worth flagging —
it means the router is moving even though end-to-end quality hasn't
visibly dropped yet, making it a useful early-warning signal
```

The shared expert (`ARCHITECTURE_V3.md` §40) is unaffected by this
diagnostic by construction — it has no routing decision to drift, since
it is always active. This makes it, incidentally, a naturally more
forgetting-resistant path for whatever "always relevant" capability you
choose to route through it (a design choice worth deliberately exploiting
if a particular capability keeps showing routing drift in practice).

## 32. Video and Document Grounded Verification

v2's core self-critique weakness — the model judging its own output
without ground truth (`SELF_LEARNING.md` §13) — extends to video and
document sessions the same way it extended to images in v2
(`SELF_LEARNING_V2.md` §17), and gets *harder* again for the same reason:
"is this a good answer" now also means "is this actually grounded in what
happened across these frames" or "is this actually grounded in what's on
this page."

**Video-grounded verification** reuses v2's frozen-encoder-plus-cached-
tokens pattern directly: checkable question types (counting events across
frames, temporal ordering — "what happened first") get a verifier that
checks the answer against ground-truth structure in the video-QA data,
where available, the same way v2's checkable image questions did. Purely
descriptive or interpretive video questions ("describe what's happening")
fall back to self-critique, with the same noise caveats v1 and v2 already
document.

**Document-grounded verification** follows the same split: checkable
questions (exact figures from a table, specific text extraction) get a
verifier that checks against the source document's actual content;
open-ended document questions ("summarize this page") fall back to
self-critique.

Both reuse `SELF_LEARNING_V2.md` §17's `image_ref`-cached-token pattern
directly, extended to the `video_ref`/`document_ref` fields from §29 —
no new verification architecture, just the existing verifier-vs-critique
split applied to two more modalities.

## 33. Self-Learning Inside Long-Horizon Tool Chains

Long-horizon tool-use chains (`ARCHITECTURE_V3.md` §46) introduce a
question v1/v2 didn't have to answer: when a multi-step chain completes,
what exactly goes into the replay buffer — the whole chain, or each step?

**v3.0's answer: each step is its own replay entry**, with the chain's
final outcome (whether the overall task succeeded) used to weight the
score of every step's entry, not just the last one. This follows the same
logic v1's existing score²-weighted sampling already uses (`SELF_LEARNING.md`
§6) — a step that was part of a chain that ultimately succeeded gets
credit propagated back to it, similar in spirit to how GRPO already
credits intermediate reasoning tokens based on final-answer correctness,
just extended across tool-call steps rather than within a single
response.

`topic: "tool-use"` entries (§29) may additionally carry a `chain_id`
field linking them back to their originating chain, purely for
diagnostic/debugging purposes — it does not change sampling or eviction
behaviour, which continues to treat each entry independently per v1's
existing policy.

## 34. Hardware Gating (v3 Additions)

Extending `SELF_LEARNING_V2.md` §19's Kaggle-only vision gate:

| Session type | Hardware | Reasoning |
|---|---|---|
| Text-only (v1, unchanged) | i3 | CPU-safe deferred-gradient mode |
| Image-grounded (v2, unchanged) | Kaggle only | Frozen ViT forward pass per turn |
| Video-grounded (v3, new) | Kaggle only | Frozen ViT forward pass × sampled frame count per turn — strictly more expensive than single-image, same gate, tighter reasoning |
| Document-grounded (v3, new) | Kaggle only | Frozen ViT forward pass × sampled page count per turn |
| Tool-use chains, text/tool-result only (v3, new) | i3 | Orchestration overhead is lightweight; follows text-only rules |
| Tool-use chains with multimodal results (v3, new) | Kaggle only | Inherits the video/document gate the moment any step's result is an image, video, or document |
| Forgetting diagnostics, text/tool-use probes (v3, new) | i3 | Same cost class as existing eval-harness text subsets |
| Forgetting diagnostics, vision/video/document probes (v3, new) | Kaggle only | Inherits the existing vision gate |
| Fine-grained MoE routing-drift diagnostic (v3, new) | Kaggle only | Requires the full MoE forward pass with routing introspection, GPU-scale batching assumed per `ARCHITECTURE_V3.md` §51 |

Same discipline as v2: a session refuses to start on ungated hardware
with a clear error message, rather than silently degrading.

## 35. Full Loop Flow (v3, All Modalities)

```
User turn arrives (text, and optionally image/video/document input,
and optionally as part of a multi-step tool chain)
        │
        ▼
Hardware gate check (§34) — refuses cleanly if session type exceeds
the current machine's capability
        │
        ▼
Model generates a response (§30's hybrid attention stack, §31's MoE
routing, §41's MTP heads available as speculative-decode draft)
        │
        ▼
Verification: checkable question -> grounded verifier (text: v1 §7,
vision: v2 §17, video/document: §32 above)
              open-ended -> self-critique (all modalities, same noise
              caveats documented since v1)
        │
        ▼
Score -> replay buffer entry (§29's schema, chain-aware per §33 where
applicable)
        │
        ▼
Committed online GRPO, deferred-gradient flush, or replay update
(SELF_LEARNING.md §5 and §8)
        │
        ▼
Forgetting diagnostics run (§27): capability probes, forgetting_delta(),
routing-drift check (§31 where MoE is enabled)
        │
        ▼
Session summary: scores, any flagged forgetting deltas, any flagged
routing drift — surfaced to you, not silently swallowed
```

## 36. CLI Commands (v3)

```
[x] aarambh-ai selflearn start ... --forgetting-manifest <manifest>
      # captures a session baseline and probes each committed update
[x] aarambh-ai selflearn flush-gradients ... --forgetting-manifest <manifest>
[x] aarambh-ai selflearn replay ... --forgetting-manifest <manifest>
[x] aarambh-ai selflearn forgetting-report --forgetting-store <curves.json>
[x] aarambh-ai eval ... --forgetting-manifest <manifest>
      # standalone named-checkpoint comparison and optional Manas JSONL export
```

## 37. Crate Structure Additions

```
crates/aarambh-ai-selflearn/
└── src/
    ├── ...v1/v2 modules unchanged (online_grpo.rs, critique.rs,
    │      replay.rs, vision_cache.rs, vision_verifier.rs, gating.rs)...
    └── forgetting_hook.rs      ← wires capability probes and MoE routing
                                  drift into the post-update-batch loop
```

All new modules are additive to the existing crate — no existing v1/v2
module is renamed, removed, or restructured.

## 38. What to Expect (v3)

- Text-only and image-only sessions behave byte-for-byte as before —
  this document changes nothing about them.
- Video and document self-learning sessions are Kaggle-only, following
  the same reasoning v2 established for images, one level more
  expensive.
- Forgetting diagnostics add a small, visible overhead per update batch
  (probe-run time), in exchange for an actual measured answer to "is the
  existing defence working," rather than an assumption.
- Fine-grained MoE sessions will occasionally surface routing-drift
  flags even when scores look fine — this is expected and is the whole
  point of the diagnostic; it is an early-warning signal, not
  necessarily evidence of a problem requiring action.
- Long-horizon tool-use chains contribute one replay entry per step, with
  credit propagated from the chain's final outcome — a single failed
  chain does not necessarily mean every step in it was a bad decision,
  and the score-weighting reflects that gradation rather than an
  all-or-nothing penalty.

## 39. Known Limitations (v3-Specific)

- Routing-drift diagnostics (§31) detect *that* routing changed, not
  *why* — distinguishing "the router legitimately learned a better
  routing decision" from "the router forgot a good routing decision"
  requires your judgment, the same way v1's self-critique noise always
  has.
- The Manas export schema (§28) is an explicit JSONL interchange contract.
  There is no automatic sync; a caller must move/import the file.
- Video/document verification (§32) is only as good as the checkable
  subset of the underlying free/public datasets — purely interpretive
  questions still fall back to self-critique with the same reliability
  caveats v1 has documented since the beginning.
- Chain-aware replay (§33) credits steps based on final chain outcome
  only; it does not yet attempt fine-grained per-step credit assignment
  (e.g. identifying which specific step in a failed chain was actually
  the mistake) — a natural v4 candidate once enough chain data exists to
  make that tractable.
