#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <student-config> <student-model> <teacher-config> <teacher-model> <tokenizer> <prompts-jsonl> <output-dir>" >&2
  exit 2
fi

STUDENT_CONFIG=$1
STUDENT_MODEL=$2
TEACHER_CONFIG=$3
TEACHER_MODEL=$4
TOKENIZER=$5
PROMPTS=$6
OUTPUT_DIR=$7
BIN=${AARAMBH_BIN:-target/release/aarambh-ai}
STEPS=${STEPS:-1000}
MAX_EPOCHS=${MAX_EPOCHS:-100000}
BATCH_SIZE=${BATCH_SIZE:-1}
ROLLOUTS=${ROLLOUTS:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
SEED=${SEED:-42}

for path in "$BIN" "$STUDENT_CONFIG" "$STUDENT_MODEL" "$TEACHER_CONFIG" "$TEACHER_MODEL" "$TOKENIZER" "$PROMPTS"; do
  [[ -e "$path" ]] || {
    echo "missing Phase 33 comparison input: $path" >&2
    exit 2
  }
done

mkdir -p "$OUTPUT_DIR"

"$BIN" distill prepare-offline \
  --teacher-config "$TEACHER_CONFIG" \
  --teacher-model "$TEACHER_MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT_DIR/offline_teacher.jsonl" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED"

"$BIN" distill train \
  --config "$STUDENT_CONFIG" \
  --student "$STUDENT_MODEL" \
  --tokenizer "$TOKENIZER" \
  --prompts "$PROMPTS" \
  --output "$OUTPUT_DIR/on_policy" \
  --teacher local \
  --teacher-config "$TEACHER_CONFIG" \
  --teacher-model "$TEACHER_MODEL" \
  --objective soft-kl \
  --rollouts-per-prompt "$ROLLOUTS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$STEPS" \
  --max-epochs "$MAX_EPOCHS"

"$BIN" distill train-offline \
  --config "$STUDENT_CONFIG" \
  --student "$STUDENT_MODEL" \
  --tokenizer "$TOKENIZER" \
  --data "$OUTPUT_DIR/offline_teacher.jsonl" \
  --output "$OUTPUT_DIR/offline" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$STEPS" \
  --max-epochs "$MAX_EPOCHS"

evaluate_case() {
  local label=$1
  local model=$2
  "$BIN" distill evaluate \
    --config "$STUDENT_CONFIG" \
    --student "$model" \
    --tokenizer "$TOKENIZER" \
    --prompts "$PROMPTS" \
    --teacher local \
    --teacher-config "$TEACHER_CONFIG" \
    --teacher-model "$TEACHER_MODEL" \
    --objective soft-kl \
    --rollouts-per-prompt "$ROLLOUTS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "$OUTPUT_DIR/$label.json" \
    --markdown "$OUTPUT_DIR/$label.md" \
    --seed "$SEED"
}

evaluate_case baseline "$STUDENT_MODEL"
evaluate_case on_policy "$OUTPUT_DIR/on_policy/final/model.safetensors"
evaluate_case offline "$OUTPUT_DIR/offline/final/model.safetensors"

echo "Phase 33 matched-update reports written to $OUTPUT_DIR"
