#!/usr/bin/env bash
# Phase 39 — OPTIONAL Kaggle helper: on-policy distillation with Max mode.
#
# This script is OPTIONAL. It documents how to run the on-policy distillation
# pipeline (Phase 33) with the new Max thinking budget so a student model
# learns to allocate thinking length on the hardest problems. It is not
# executed by CI or by scripts/phase39_smoke.sh.
#
# Prerequisites (all must be supplied by the operator):
#   * A Kaggle (or equivalent GPU) environment with the workspace checked out.
#   * PHASE39_STUDENT       — student checkpoint (safetensors).
#   * PHASE39_TOKENIZER     — tokenizer JSON.
#   * PHASE39_CONFIG        — distill smoke config TOML.
#   * PHASE39_TEACHER_MODEL — frozen teacher checkpoint (safetensors).
#   * PHASE39_TEACHER_CONFIG — teacher training config TOML.
#   * PHASE39_PROMPTS       — prompts JSONL.
#
# Distillation reuses the existing rollout + objective logic unchanged;
# Max only enlarges the student thinking budget during rollout collection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PHASE39_STUDENT:?PHASE39_STUDENT is required}"
: "${PHASE39_TOKENIZER:?PHASE39_TOKENIZER is required}"
: "${PHASE39_CONFIG:?PHASE39_CONFIG is required}"
: "${PHASE39_TEACHER_MODEL:?PHASE39_TEACHER_MODEL is required}"
: "${PHASE39_TEACHER_CONFIG:?PHASE39_TEACHER_CONFIG is required}"
: "${PHASE39_PROMPTS:?PHASE39_PROMPTS is required}"

OUTPUT="${PHASE39_OUTPUT:-adapters/distill-max}"
ROLLOUTS="${PHASE39_ROLLOUTS:-4}"
MAX_NEW_TOKENS="${PHASE39_MAX_NEW_TOKENS:-1024}"

echo "==> Phase 39 distillation with Max thinking budget (optional, Kaggle)"
cargo run --quiet --locked --release -p aarambh-studio -- distill train \
  --config "$PHASE39_CONFIG" \
  --student "$PHASE39_STUDENT" \
  --tokenizer "$PHASE39_TOKENIZER" \
  --prompts "$PHASE39_PROMPTS" \
  --output "$OUTPUT" \
  --teacher local \
  --teacher-model "$PHASE39_TEACHER_MODEL" \
  --teacher-config "$PHASE39_TEACHER_CONFIG" \
  --rollouts-per-prompt "$ROLLOUTS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --thinking max

echo "Phase 39 Kaggle distillation helper completed; adapter saved to $OUTPUT"
echo "(Optional script — not run by CI.)"
