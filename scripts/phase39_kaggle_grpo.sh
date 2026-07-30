#!/usr/bin/env bash
# Phase 39 — OPTIONAL Kaggle helper: GRPO re-run including Max-budget rollouts.
#
# This script is OPTIONAL. It is a documented starting point for running the
# Stage 2 GRPO re-run described in ROADMAP_V3.md (Phase 39) on Kaggle GPU
# hardware. It is not executed by CI or by scripts/phase39_smoke.sh.
#
# Prerequisites (all must be supplied by the operator):
#   * A Kaggle (or equivalent GPU) environment with the workspace checked out.
#   * PHASE39_BASE     — trainable base checkpoint (safetensors).
#   * PHASE39_REFERENCE — frozen reference checkpoint (safetensors).
#   * PHASE39_TOKENIZER — tokenizer JSON.
#   * PHASE39_CONFIG    — training config TOML.
#   * PHASE39_DATA      — GRPO JSONL prompt/data file.
#
# Max mode reuses the existing format verifier and reward shaping unchanged
# (ARCHITECTURE_V3.md §48.2): no separate reward function, only a larger
# budget ceiling within the same incentive structure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PHASE39_BASE:?PHASE39_BASE (trainable base checkpoint) is required}"
: "${PHASE39_REFERENCE:?PHASE39_REFERENCE (frozen reference checkpoint) is required}"
: "${PHASE39_TOKENIZER:?PHASE39_TOKENIZER is required}"
: "${PHASE39_CONFIG:?PHASE39_CONFIG is required}"
: "${PHASE39_DATA:?PHASE39_DATA (GRPO prompt data) is required}"

OUTPUT="${PHASE39_OUTPUT:-adapters/grpo-max}"
GROUP_SIZE="${PHASE39_GROUP_SIZE:-8}"
MAX_NEW_TOKENS="${PHASE39_MAX_NEW_TOKENS:-1024}"
STEPS="${PHASE39_STEPS:-200}"

echo "==> Phase 39 GRPO re-run with Max-budget rollouts (optional, Kaggle)"
cargo run --quiet --locked --release -p aarambh-studio -- finetune grpo \
  --config "$PHASE39_CONFIG" \
  --base "$PHASE39_BASE" \
  --reference "$PHASE39_REFERENCE" \
  --tokenizer "$PHASE39_TOKENIZER" \
  --data "$PHASE39_DATA" \
  --output "$OUTPUT" \
  --verifier math-format \
  --group-size "$GROUP_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --thinking max \
  --steps "$STEPS"

echo "Phase 39 Kaggle GRPO helper completed; adapter saved to $OUTPUT"
echo "(Optional script — not run by CI.)"
