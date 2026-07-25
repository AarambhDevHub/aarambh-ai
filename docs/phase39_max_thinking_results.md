# Phase 39 — Max Thinking Mode

**Version:** 3.0.0
**Status:** Implemented. Max mode is a fifth `ThinkingMode` variant, not a new
reasoning algorithm — it reuses every existing reasoning mechanism with a
larger thinking budget.

---

## 1. What changed

Aarambh-AI's thinking system already supported four budget modes:

| Mode   | Nominal budget |
|--------|---------------:|
| `none`   | 0              |
| `low`    | 256            |
| `medium` | 1,024          |
| `high`   | 4,096          |

Phase 39 adds a fifth mode, `max`, with a **16,384-token** nominal budget — the
next step in the same ~4x progression the existing four modes already follow
(0 → 256 → 1,024 → 4,096 → 16,384).

Max is **only** a larger budget value. It introduces:

- **No** new reasoning algorithm.
- **No** structural change to `ThinkingController`. The same `ForceOpen`
  (`<think>`) / `ForceClose` (`</think>`) forced-token mechanism, the same
  budget tracking, and the same collapse-on-force-close behaviour every
  existing mode already has.
- **No** change to runtime budget clamping. The effective thinking budget
  still respects `max_new_tokens`, the answer reserve, and the model
  `max_seq_len`.
- **No** change to GRPO reward shaping or distillation objectives.

### Centralised parsing and display

Thinking-mode parsing and display were centralised on `ThinkingMode` itself
(`std::str::FromStr` + `std::fmt::Display`). Every CLI command, the serving
API, GRPO, and distillation now share one canonical
`none | low | medium | high | max` vocabulary. The previously-duplicated
string-match parsers in `infer`, `serve`, and the server now delegate to
`ThinkingMode::from_str`. `GrpoThinkingMode` keeps its own mirror of the
vocabulary because the `aarambh-ai-finetune` crate sits below
`aarambh-ai-inference` in the dependency layering and therefore cannot depend
on it; its `ThinkingMode`↔`GrpoThinkingMode` conversions are owned by the
self-learning crate, which depends on both, so no parsing logic is duplicated
across crate boundaries.

### Sampling defaults

`ThinkingMode::default_sampler()` extends v1's per-mode table
(`ARCHITECTURE_V3.md` §48.3):

| Mode   | Temperature | Top-p |
|--------|------------:|------:|
| `none`   | 0.70        | 0.90  |
| `low`    | 0.75        | 0.92  |
| `medium` | 0.80        | 0.95  |
| `high`   | 0.80        | 0.95  |
| `max`    | **0.85**    | **0.97** |

Max is the most exploratory of the five: Max-mode tasks are exactly the ones
where premature convergence on a wrong early step is most costly. The serving
API applies these defaults **only** when the caller omits `temperature`/
`top_p`; explicit user parameters are never overridden.

---

## 2. Commands

Max mode is accepted everywhere High mode is accepted:

```bash
# Inference
aarambh-ai infer --config configs/tiny_shakespeare.toml \
  --thinking max --prompt "Prove that the sum of two odd integers is even."

# Agent (long-horizon tool chains — pairs with Phase 37)
aarambh-ai agent --config configs/tiny_shakespeare.toml \
  --thinking max --tools tools.json --prompt "Plan and execute ..."

# Evaluation (Phase 39 hard-problems task)
aarambh-ai eval --config configs/tiny_shakespeare.toml \
  --tasks hard-problems --thinking max --max-new-tokens 512

# Serving (OpenAI-compatible reasoning_effort)
aarambh-ai serve --thinking max
# or per-request:  "reasoning_effort": "max"

# Fine-tuning (GRPO)
aarambh-ai finetune grpo --config configs/tiny_shakespeare.toml \
  --thinking max --base model.safetensors --reference ref.safetensors ...

# Distillation
aarambh-ai distill train --config configs/distill_smoke.toml \
  --thinking max --student student.safetensors ...

# Self-learning
aarambh-ai selflearn start --thinking max --prompt "..."
```

Invalid values are rejected by the shared parser:

```
invalid thinking mode 'ultra', expected none|low|medium|high|max
```

---

## 3. Expected outputs

### `infer --thinking max`

The thinking block is forced open, content is budget-tracked up to the
effective budget (clamped to `max_new_tokens` minus the answer reserve), and
the block is force-closed — identical to High mode, only with a larger
ceiling. Completion metadata reports the same fields as every other mode.

### `eval --tasks hard-problems --thinking max`

The `hard-problems` task reports accuracy plus average token accounting in the
scorecard `details` map:

```json
{
  "name": "hard-problems",
  "metric": "accuracy",
  "value": 0.625,
  "examples": 8,
  "correct": 5,
  "details": {
    "thinking_tokens": 412.5,
    "completion_tokens": 38.25,
    "total_tokens": 450.75
  }
}
```

`thinking_tokens` counts content tokens inside the `<think>` block (excluding
markers), `completion_tokens` counts answer tokens, and `total_tokens` is
their sum. Generation is deterministic (greedy argmax) regardless of mode.

### `serve` with `reasoning_effort: "max"`

The server resolves the thinking mode, applies Max's sampling defaults
(temperature 0.85, top_p 0.97) when the caller omits them, and runs the same
`ThinkingController` path. Unknown `reasoning_effort` values are rejected with
a 400 and the canonical error message.

---

## 4. High-vs-Max comparison

The point of Phase 39 is whether Max earns its larger budget on problems where
High's 4,096-token ceiling was previously insufficient
(`ARCHITECTURE_V3.md` §48.4). The `hard-problems` fixture holds deterministic
multi-step problems selected for this comparison.

Run both modes and compare:

```bash
for mode in high max; do
  aarambh-ai eval --config configs/tiny_shakespeare.toml \
    --model model.safetensors --tasks hard-problems \
    --max-new-tokens 512 --thinking "$mode" \
    --out scorecard-${mode}.json
done
```

### Comparison table (schema)

| Mode | Accuracy | Avg thinking tokens | Avg completion tokens | Avg total tokens |
|------|---------:|--------------------:|----------------------:|-----------------:|
| high | _measured_ | _measured_ | _measured_ | _measured_ |
| max  | _measured_ | _measured_ | _measured_ | _measured_ |

> **Note:** This table is intentionally a schema. Aarambh-AI ships **no
> pretrained checkpoints**, so no benchmark numbers are reported here. The
> `hard-problems` fixture is deterministic, so once a trained checkpoint is
> supplied the table can be filled by running the commands above. The
> `scripts/phase39_smoke.sh` and `scripts/phase39_kaggle_compare.sh` scripts
> produce the two scorecards and a printed comparison line automatically.

The validation criterion (`ROADMAP_V3.md` Phase 39 milestone): Max should
spend at least as many tokens as High and improve accuracy on the
High-insufficient holdout. A larger budget that does not measurably improve
accuracy on the problems it was built for is not a validated feature.

---

## 5. Kaggle results (placeholder)

The heavier GRPO re-run and full High-vs-Max comparison run on Kaggle GPU
hardware. Three **optional** helper scripts document the workflow; they are
not executed by CI and require operator-supplied checkpoints:

- `scripts/phase39_kaggle_grpo.sh` — Stage 2 GRPO re-run including
  Max-budget rollouts. Reuses the existing format verifier and reward shaping
  unchanged.
- `scripts/phase39_kaggle_distill.sh` — on-policy distillation with the Max
  thinking budget during student rollout collection. Reuses existing rollout
  and objective logic.
- `scripts/phase39_kaggle_compare.sh` — High-vs-Max `hard-problems`
  comparison writing both scorecards and a printed summary line.

> **No benchmark numbers are reported here.** Kaggle results are placeholders
> to be filled in by an operator with trained checkpoints; the scripts only
> produce the measurement, they do not assert specific accuracy figures.

---

## 6. Tests

Phase 39 adds unit and integration tests covering:

- `ThinkingMode::Max` budget (16,384) and `is_enabled()`.
- The centralised `FromStr` parser (all five modes, case-insensitive, rejects
  unknowns) and `Display` (canonical lower-case, round-trips through parser).
- `ThinkingController` force-closing Max at budget exactly like other modes
  (no special-cased logic path).
- `ThinkingMode::default_sampler()` per-mode table, with Max strictly more
  exploratory than High.
- `GrpoThinkingMode` and `DistillThinkingMode` parsing Max (and the distill
  test that previously asserted `from_str("max").is_err()` is flipped).
- Self-learning `thinking_to_grpo` / `grpo_to_thinking` mapping Max.
- The `hard-problems` task fixture parsing and stable name.
- The `phase39_smoke.sh` smoke script (source/contract checks always;
  checkpoint execution gated on `PHASE39_MODEL`).

Run them with:

```bash
cargo test --workspace
scripts/phase39_smoke.sh
```

---

## 7. Regression guarantee

`None`/`Low`/`Medium`/`High` behaviour is byte-for-byte unchanged after Max is
added — covered by the `existing_none_low_medium_high_modes_are_byte_for_byte_unchanged`
regression test and by the unchanged `ThinkingController` mechanism that all
five modes share.
