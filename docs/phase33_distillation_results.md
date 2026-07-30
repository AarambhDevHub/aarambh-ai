# Phase 33 On-Policy Distillation

Phase 33 trains a student on completions sampled from its current weights and
scored by a frozen teacher. This document defines the data contracts, execution
proof, and the matched comparison required before making a quality claim.

## Implemented Pipeline

1. The student prefills each prompt once and forks that cache into independently
   seeded rollout sessions.
2. The existing inference engine batches autoregressive decode. Generation has
   no gradient graph.
3. A local checkpoint teacher returns detached packed token logits and scalar
   mean log-probability scores, or a scored-reference teacher returns weighted
   token-F1 rewards and an optional correction.
4. Replay runs the student on prompt plus generated completion. Packed indices
   select only non-forced completion targets.
5. Forward KL or group-normalized reward-policy loss is combined with configured
   MTP, MoE balance, and periodic DSA indexer losses.
6. Full student weights update through AdamW. Checkpoints preserve optimizer
   moments and the exact deterministic prompt cursor/order.

The scored-reference schema is JSONL:

```json
{"id":"math-1","prompt":"Two plus two is","references":[{"completion":" four","score":1.0}]}
```

The local teacher and student must share a tokenizer and output vocabulary, but
their hidden dimensions, layer counts, attention schedules, and MoE layouts may
differ.

## CPU Execution Proof

Build an MTP smoke checkpoint, then run every Phase 33 mode:

```sh
cargo build --release -p aarambh-studio
target/release/aarambh-studio train --config configs/mtp_smoke.toml
scripts/phase33_smoke.sh
```

The smoke performs two local soft-KL optimizer steps, one scored-reference
reward step, static offline-teacher generation, one offline optimizer step,
fresh-rollout evaluation, numbered/final checkpoint writes, and exact resume.
It is an execution and finite-gradient proof only. Four prompts and two updates
cannot establish model-quality improvement.

## Kaggle Recipe

Create deterministic prompt JSONL from WikiText-103 or another licensed local
corpus:

```sh
python3 scripts/phase33_prepare_prompts.py \
  data/wikitext-103-raw/wiki.train.raw \
  data/distill_prompts.jsonl \
  --max-prompts 10000 --max-chars 512
```

Run a Medium student against a larger local teacher by supplying the matching
teacher architecture config and checkpoint:

```sh
target/release/aarambh-studio distill train \
  --config configs/medium_distill.toml \
  --student checkpoints/wikitext103_medium_mtp/best/model.safetensors \
  --tokenizer checkpoints/wikitext103_medium/tokenizer.json \
  --prompts data/distill_prompts.jsonl \
  --output checkpoints/wikitext103_medium_distill \
  --teacher local \
  --teacher-config configs/large_distill.toml \
  --teacher-model checkpoints/wikitext103_large_mtp/best/model.safetensors \
  --objective soft-kl \
  --rollouts-per-prompt 4 \
  --max-new-tokens 128
```

Use `--teacher dataset --teacher-data <references.jsonl> --objective reward`
when memory cannot hold a local teacher. Reward mode requires at least two
rollouts per prompt so advantages can be normalized within each prompt group.

## Matched Comparison

The comparison script starts both students from the same checkpoint, uses the
same prompt set and optimizer-step budget, freezes teacher completions for the
offline control, and scores baseline/on-policy/offline checkpoints on fresh
student rollouts with the same teacher and sampling seed:

```sh
STEPS=10000 ROLLOUTS=4 MAX_NEW_TOKENS=128 \
scripts/phase33_compare_distillation.sh \
  configs/medium_distill.toml \
  checkpoints/wikitext103_medium_mtp/best/model.safetensors \
  configs/large_distill.toml \
  checkpoints/wikitext103_large_mtp/best/model.safetensors \
  checkpoints/wikitext103_medium/tokenizer.json \
  data/distill_prompts.jsonl \
  reports/phase33
```

The generated `baseline.json`, `on_policy.json`, and `offline.json` report mean
teacher reward, teacher-to-student forward KL, completion length, token count,
and end-to-end scoring throughput.

## Acceptance Rule

A Phase 33 quality win is recorded only when the held-out on-policy checkpoint:

- has lower fresh-rollout teacher-to-student KL than the matched offline model;
- does not regress the repository eval scorecard beyond its configured
  significance threshold; and
- reproduces across at least three seeds with the prompt split and hardware
  recorded alongside the reports.

No Medium/Large Kaggle result is checked into this source tree yet, so this
repository makes no unmeasured quality or speedup claim.
