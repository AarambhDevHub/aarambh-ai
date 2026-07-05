# SELF_LEARNING_V2.md — aarambh-ai v2.0

> Companion to `SELF_LEARNING.md`. This document covers **only what v2.0
> adds** — vision-aware self-learning. Sections continue numbering from
> v1's Section 13. Everything in `SELF_LEARNING.md` (Online GRPO,
> Self-Critique, Experience Replay, CPU vs GPU mode, catastrophic
> forgetting protection) is unchanged and continues to work exactly as
> documented there for text-only sessions.

---

## Table of Contents

14. [Why This Is a Separate Document](#14-why-this-is-a-separate-document)
15. [What Changes, What Doesn't](#15-what-changes-what-doesnt)
16. [Extended Replay Buffer Schema](#16-extended-replay-buffer-schema)
17. [Vision-Grounded Verification](#17-vision-grounded-verification)
18. [Extended Online GRPO](#18-extended-online-grpo)
19. [Hardware Gating: Why Kaggle-Only](#19-hardware-gating-why-kaggle-only)
20. [Full Loop Flow (Vision Mode)](#20-full-loop-flow-vision-mode)
21. [CLI Commands (Vision Mode)](#21-cli-commands-vision-mode)
22. [Crate Structure Additions](#22-crate-structure-additions)
23. [What to Expect (Vision Mode)](#23-what-to-expect-vision-mode)
24. [Known Limitations (Vision-Specific)](#24-known-limitations-vision-specific)

---

## 14. Why This Is a Separate Document

Vision support (`ARCHITECTURE_V2.md` §24–25) changes self-learning in ways
that are substantial enough to warrant their own design pass rather than a
few extra lines in the existing loop:

- The replay buffer needs to store an image reference alongside every
  entry that came from a visual turn, without breaking the plain-text
  buffers every existing text-only session already has on disk.
- Self-critique's core weakness — the model judging its own output without
  ground truth (`SELF_LEARNING.md` §13) — gets *harder*, not just noisier,
  once "is this a good answer" also means "is this actually grounded in
  what's in the image."
- The frozen ViT encoder's forward pass adds real memory and time cost to
  *every single turn*, which changes the CPU-safe-mode calculus from v1
  §7 enough that vision self-learning is scoped to Kaggle only.

## 15. What Changes, What Doesn't

**Unchanged, and verified unchanged by regression tests:**
- Text-only self-learning on i3 — same CPU-safe deferred-gradient mode,
  same replay buffer behaviour, same critique prompt, same eviction
  policy, same catastrophic-forgetting protections (`SELF_LEARNING.md`
  §8), byte-for-byte.
- Existing `replay_buffer.jsonl` files load exactly as before. No
  migration step is required for text-only users.

**New, additive only:**
- An optional `image_ref` field on replay entries.
- A vision-grounded verifier for checkable question types, used *instead
  of* self-critique where it applies — self-critique remains the fallback
  for open-ended image description, with the same noise caveats v1
  already documents for text.
- A hardware gate that refuses to start a vision self-learning session on
  i3, with a clear error message, rather than silently degrading.

## 16. Extended Replay Buffer Schema

v1's replay entries (`SELF_LEARNING.md` §6) are plain JSONL:

```json
{"prompt": "What is recursion?", "response": "Recursion is when...", "score": 0.84, "timestamp": 1719000000, "topic": "code"}
```

v2.0 adds one optional field:

```json
{"prompt": "How many cats are in this image?", "response": "There are 3 cats.", "score": 0.95, "timestamp": 1719000000, "topic": "vision", "image_ref": "cache/img_a1b2c3.safetensors"}
```

`image_ref` points to a **cached vision embedding**, not the raw image
file — the frozen ViT encoder's output for that image, saved once at
first use. This matters for replay specifically: replay fine-tuning
resamples the same entries repeatedly, and re-running the frozen encoder
forward pass on every replay batch would be pure wasted compute for a
component that never changes. Caching the embedding once and reusing it
is both faster and exactly correct, since the encoder is frozen.

**Schema versioning:** the buffer file gains a `replay_buffer_v2.jsonl`
name for sessions that include any vision entries. Existing
`replay_buffer.jsonl` (v1, text-only) files load unchanged, with
`image_ref` defaulting to `None` for every entry — this is enforced by a
regression test (`v1_replay_buffer_loads_with_image_ref_defaulting_to_none`).

**Eviction and sampling** (`SELF_LEARNING.md` §6) are unchanged — score²-
weighted sampling and the diversity constraint apply identically, with
`"vision"` simply added as a new topic alongside `math, code, reasoning,
factual, creative, general`.

## 17. Vision-Grounded Verification

### The problem with self-critique on images

v1's self-critique (`SELF_LEARNING.md` §5) works by prompting the model to
score its own prior response — a reasonable proxy when there's no ground
truth to check against, but already documented as noisy for text
(`SELF_LEARNING.md` §13). For images, the failure mode is worse: a small
model can produce a fluent, plausible-sounding description that is
*confidently wrong* about what's actually in the image, and then score its
own hallucination highly, because the critique step doesn't re-examine the
image any more carefully than the original response did.

### The fix: verify what's checkable, critique what isn't

Rather than trusting self-critique for all visual turns, v2.0 splits
vision questions into two classes, mirroring the existing split between
GRPO's deterministic verifiers and self-critique's fallback role
(`SELF_LEARNING.md` §13):

| Question type | Example | Verification method |
|---|---|---|
| Checkable | "How many dogs are in this image?" | `VisionVerifier` — exact match against a known count/label/presence check |
| Checkable | "Is there a red car in this image?" | `VisionVerifier` — yes/no against known presence |
| Open-ended | "Describe this image" | Self-critique (fallback, same noise caveats as v1 text) |

`VisionVerifier` follows the same trait shape as `MathVerifier` and
`CodeVerifier` (`ARCHITECTURE.md` §12.1, reused directly in
`aarambh-ai-eval`'s GSM8K/HumanEval tasks) — it needs ground truth to check
against, which for self-learning means the question set has to come with
known answers (e.g. counting/colour/presence questions derived from a
labelled subset, not arbitrary user-uploaded images with no known answer).

**This is a real constraint, not a workaround:** for genuinely novel
images the user provides with no known answer, there is no ground truth to
verify against, and the loop falls back to self-critique with the same
honest caveat v1 gives for open-ended text — it filters the worst errors
via the score threshold, but some incorrect entries will still be stored.

## 18. Extended Online GRPO

`OnlineGrpo` (`SELF_LEARNING.md` §4) gains an optional vision verifier
slot:

```rust
struct OnlineGrpoConfig {
    // ...all v1 fields unchanged...
    vision_verifier: Option<Box<dyn VisionVerifier>>,
}
```

When `vision_verifier` is `None` (the default, and the only state possible
for a text-only session), behaviour is identical to v1 — this is enforced
by a regression test
(`text_only_self_learn_path_is_unmodified_by_this_phase`). When present,
checkable vision questions route through it instead of self-critique,
following the same generate → score → advantage-weighted update → KL
penalty loop already documented in `SELF_LEARNING.md` §4 — only the
scoring step's ground truth source changes.

## 19. Hardware Gating: Why Kaggle-Only

v1's i3 self-learning overhead is documented at ~400 MB peak
(`SELF_LEARNING.md` §7, `ARCHITECTURE.md` §18) — comfortable on an 8 GB
machine. Adding a frozen ViT encoder forward pass (~344 MB weights alone,
`ARCHITECTURE_V2.md` §33) to *every single turn* changes that math
substantially: it's not that the i3 categorically cannot run a ViT forward
pass, it's that doing so on every self-learning turn, on top of the
existing generation + critique + gradient-accumulation overhead, pushes
well past what's comfortable to promise as "just works" on an 8 GB laptop
already running everything else.

Rather than let vision self-learning silently degrade into swap-thrashing
or OOM on i3, v2.0 gates it explicitly:

```rust
fn require_hardware(hardware: Hardware) -> Result<()> {
    if hardware == Hardware::I3Cpu {
        return Err(AarambhError::UnsupportedHardware(
            "Vision self-learning requires Kaggle GPU. \
             Text-only self-learning remains fully supported on i3 — \
             see SELF_LEARNING.md §7."
        ));
    }
    Ok(())
}
```

`aarambh-ai selflearn start --mode vision` on i3 fails fast with this
message rather than attempting to run. `--mode text` (the default,
unchanged from v1) is unaffected and continues to work exactly as
`SELF_LEARNING.md` describes.

## 20. Full Loop Flow (Vision Mode)

Extends `SELF_LEARNING.md` §14.6's flow diagram for vision turns only —
text-only turns within the same session follow the unmodified v1 flow.

```
User provides prompt + image
         │
         ▼
Vision encoder (frozen) → image embeddings
  (cached: if this exact image was seen before, reuse the cached
   embedding instead of re-running the encoder — §16)
         │
         ▼
Projector (frozen at inference time — training only in Phase 20's
           dedicated VLM tuning, not during self-learning) → image tokens
         │
         ▼
Fusion: image tokens spliced into prompt sequence (ARCHITECTURE_V2.md §24)
         │
         ▼
Model generates N completions (same generate-N-then-score shape as v1 §4)
         │
         ▼
   ┌─────────────┴─────────────┐
   │                            │
Checkable question?        Open-ended question?
   │                            │
VisionVerifier              Self-critique
(exact-match scoring)       (fallback, same noise
   │                         caveats as v1 text)
   └─────────────┬─────────────┘
                 ▼
     Advantage-weighted GRPO update (LoRA/DoRA adapter only)
     + KL penalty vs frozen reference (same as v1 §4, §8)
                 ▼
     High-scoring entries → replay buffer (image_ref cached, §16)
                 ▼
     Periodic replay fine-tune (same SFT-on-replay mechanism as v1 §6,
     reusing cached embeddings — no repeated encoder forward passes)
```

## 21. CLI Commands (Vision Mode)

```sh
# Start a vision-capable self-learning session (Kaggle only)
aarambh-ai selflearn start --mode vision --config configs/selflearn_vision.toml

# Text-only session — unchanged from v1, works on i3
aarambh-ai selflearn start --mode text --config configs/selflearn_cpu.toml

# Vision-specific stats (extends v1's `selflearn stats` output)
aarambh-ai selflearn stats --mode vision
# Example output:
# Replay buffer: 214 / 500 entries  avg score: 0.77
# Vision (checkable):   ↑ +0.14 (last 50 vs first 50)
# Vision (open-ended):  ↑ +0.04  ← self-critique fallback, noisier signal
# Reasoning:            ↑ +0.11
# Factual:               → +0.01

# Attempting vision mode on i3 fails clearly:
$ aarambh-ai selflearn start --mode vision --config configs/selflearn_cpu.toml
error: Vision self-learning requires Kaggle GPU. Text-only self-learning
       remains fully supported on i3 — see SELF_LEARNING.md §7.
```

## 22. Crate Structure Additions

```
crates/aarambh-ai-selflearn/
└── src/
    ├── ...lib.rs, config.rs, learning_loop.rs, online_grpo.rs,
    │     critique.rs, replay.rs, metrics.rs — all unchanged from v1...
    ├── replay_buffer.rs   ← NEW: v2 schema extension (image_ref), v1-compatible load path
    ├── vision_verifier.rs ← NEW: VisionVerifier trait + checkable-question implementations
    ├── online_grpo.rs     ← EXTENDED: optional vision_verifier field on OnlineGrpoConfig
    └── gating.rs          ← NEW: require_hardware() guard for vision-mode sessions
```

**New dependency (this crate only):** `aarambh-ai-vision`
(`ARCHITECTURE_V2.md` §24), for the frozen encoder + projector used to
produce image embeddings during generation.

**Still does NOT depend on:** `aarambh-ai-safety`, unchanged from v1 — the
safety layer continues to apply at the binary level, not inside
`aarambh-ai-selflearn`, for vision sessions exactly as it does for text.

## 23. What to Expect (Vision Mode)

Mirrors the shape of `SELF_LEARNING.md` §12, with vision-specific timing.

### Turn 1–50 (early)
Encoder embeddings get cached as new images are seen. Checkable-question
scores are noisy but meaningful (real ground truth, small model). Open-
ended scores are noisy in the same way v1 §12 describes for text — self-
critique hasn't accumulated enough replay signal yet.

### Turn 50–200 (buffer filling)
Cache hit rate rises as repeated/similar images get reused. Checkable-
question accuracy trend becomes visible in `selflearn stats`.

### Turn 200+ (first Kaggle-scale flush)
Replay fine-tune runs against cached embeddings — no repeated encoder
forward passes, so this stays close to v1's replay-fine-tune cost
(`SELF_LEARNING.md` §6) despite the added modality.

### Steady state
Checkable vision questions (counting, colour, presence) show the clearest
improvement trend, same as verifiable math/code questions do in v1's
text-only loop (`SELF_LEARNING.md` §12). Open-ended image description
improves more slowly and less reliably, consistent with self-critique's
documented limitations.

## 24. Known Limitations (Vision-Specific)

Extends `SELF_LEARNING.md` §13's limitations list — all of v1's text
limitations still apply to text turns unchanged.

**Grounded verification only covers checkable question types.** Counting,
colour, and presence questions can be verified against known answers.
Open, subjective, or free-form questions about a user's own uploaded
images have no ground truth to check against, and fall back to
self-critique with the same noise already documented for text.

**Cached embeddings assume a frozen encoder.** If the vision encoder is
ever fine-tuned in a future version, every cached embedding in the replay
buffer becomes stale and must be recomputed. As long as the encoder stays
frozen (the v2.0 design, `ARCHITECTURE_V2.md` §24), this is a non-issue.

**Kaggle-only means self-learning sessions cannot span both modes on one
machine mid-session.** A user on i3 who wants to add images to an ongoing
text self-learning session needs to move that session to Kaggle — there is
no partial/degraded vision mode on i3 by design (§19), rather than a
silent quality cliff.

**Small model quality ceiling applies doubly here.** v1 §13 already notes
a 25M model's fundamental quality ceiling on text. Grounding that same
small model in visual detail is a harder task, not an easier one — expect
the vision quality ceiling to be reached faster, and be lower, than the
text-only ceiling at the same parameter count. Small (117M) or larger is
recommended for any vision self-learning where quality matters, mirroring
v1's existing recommendation for text.

**No distributed replay, unchanged from v1.** The replay buffer (with or
without image entries) remains a single JSONL file — still a single-user,
single-process system by design.
