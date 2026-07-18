#!/usr/bin/env bash
set -euo pipefail

BIN=${AARAMBH_BIN:-target/release/aarambh-ai}
CONFIG=${DISTILL_CONFIG:-configs/distill_smoke.toml}
STUDENT=${DISTILL_STUDENT:-checkpoints/mtp_smoke/step_000001/model.safetensors}
TEACHER=${DISTILL_TEACHER:-checkpoints/mtp_smoke/step_000002/model.safetensors}
TOKENIZER=${DISTILL_TOKENIZER:-checkpoints/mtp_smoke/tokenizer.json}
PROMPTS=${DISTILL_PROMPTS:-data/distill_smoke_prompts.jsonl}
SCORED_TEACHER=${DISTILL_SCORED_TEACHER:-data/distill_smoke_teacher.jsonl}
OUTPUT_ROOT=${DISTILL_OUTPUT_ROOT:-checkpoints/phase33_smoke}

for path in "$BIN" "$CONFIG" "$STUDENT" "$TEACHER" "$TOKENIZER" "$PROMPTS" "$SCORED_TEACHER"; do
  [[ -e "$path" ]] || {
    echo "missing Phase 33 smoke input: $path" >&2
    echo "build the release binary and run configs/mtp_smoke.toml first" >&2
    exit 2
  }
done

mkdir -p "$OUTPUT_ROOT"

"$BIN" distill train \
  --config "$CONFIG" \
  --student "$STUDENT" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT_ROOT/on_policy_local" \
  --teacher local \
  --teacher-model "$TEACHER" \
  --teacher-config "$CONFIG" \
  --objective soft-kl \
  --rollouts-per-prompt 2 \
  --max-new-tokens 8 \
  --batch-size 1 \
  --max-steps 2 \
  --max-epochs 2 \
  --save-every-n-steps 1 \
  --log-every-n-steps 1 \
  --no-shuffle

"$BIN" distill train \
  --config "$CONFIG" \
  --student "$STUDENT" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT_ROOT/on_policy_local" \
  --teacher local \
  --teacher-model "$TEACHER" \
  --teacher-config "$CONFIG" \
  --objective soft-kl \
  --rollouts-per-prompt 2 \
  --max-new-tokens 8 \
  --batch-size 1 \
  --max-steps 2 \
  --max-epochs 2 \
  --save-every-n-steps 1 \
  --log-every-n-steps 1 \
  --no-shuffle \
  --resume

"$BIN" distill train \
  --config "$CONFIG" \
  --student "$STUDENT" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT_ROOT/on_policy_reward" \
  --teacher dataset \
  --teacher-data "$SCORED_TEACHER" \
  --objective reward \
  --rollouts-per-prompt 2 \
  --max-new-tokens 8 \
  --batch-size 1 \
  --max-steps 1 \
  --max-epochs 1 \
  --save-every-n-steps 1 \
  --log-every-n-steps 1 \
  --no-shuffle

"$BIN" distill prepare-offline \
  --teacher-config "$CONFIG" \
  --teacher-model "$TEACHER" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT_ROOT/offline_teacher.jsonl" \
  --max-new-tokens 8 \
  --seed 42

"$BIN" distill train-offline \
  --config "$CONFIG" \
  --student "$STUDENT" \
  --tokenizer "$TOKENIZER" \
  --data "$OUTPUT_ROOT/offline_teacher.jsonl" \
  --output "$OUTPUT_ROOT/offline" \
  --batch-size 1 \
  --max-steps 1 \
  --max-epochs 1 \
  --save-every-n-steps 1 \
  --log-every-n-steps 1 \
  --no-shuffle

"$BIN" distill evaluate \
  --config "$CONFIG" \
  --student "$OUTPUT_ROOT/on_policy_local/final/model.safetensors" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --teacher local \
  --teacher-model "$TEACHER" \
  --teacher-config "$CONFIG" \
  --objective soft-kl \
  --rollouts-per-prompt 2 \
  --max-new-tokens 8 \
  --max-prompts 2 \
  --out "$OUTPUT_ROOT/eval.json" \
  --markdown "$OUTPUT_ROOT/eval.md" \
  --seed 42

echo "Phase 33 smoke completed: $OUTPUT_ROOT"
