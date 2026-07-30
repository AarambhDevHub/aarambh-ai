#!/usr/bin/env bash
# Phase 39 — OPTIONAL Kaggle helper: High-vs-Max comparison on hard problems.
#
# This script is OPTIONAL. It runs the same `hard-problems` eval task under
# High and Max thinking budgets side by side and writes both scorecards so the
# accuracy/token-spend trade-off can be compared directly. It is not executed
# by CI or by scripts/phase39_smoke.sh (the lightweight version of this
# comparison lives in scripts/phase39_smoke.sh).
#
# Prerequisites (all must be supplied by the operator):
#   * A Kaggle (or equivalent GPU) environment with the workspace checked out.
#   * PHASE39_MODEL     — checkpoint to evaluate (safetensors).
#   * PHASE39_TOKENIZER — tokenizer JSON.
#   * PHASE39_CONFIG    — training config TOML.
#
# Per ARCHITECTURE_V3.md §48.4, a larger budget that does not measurably
# improve accuracy on the problems it was built for is not a validated
# feature; this helper produces the evidence for that decision.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PHASE39_MODEL:?PHASE39_MODEL is required}"
: "${PHASE39_TOKENIZER:?PHASE39_TOKENIZER is required}"
: "${PHASE39_CONFIG:?PHASE39_CONFIG is required}"

STATE="${PHASE39_STATE:-${TMPDIR:-/tmp}/aarambh-phase39-kaggle-$$}"
mkdir -p "$STATE"
MAX_NEW_TOKENS="${PHASE39_MAX_NEW_TOKENS:-512}"
MAX_EXAMPLES="${PHASE39_MAX_EXAMPLES:-}"

EXTRA=()
if [[ -n "$MAX_EXAMPLES" ]]; then
  EXTRA+=(--max-examples "$MAX_EXAMPLES")
fi

echo "==> Phase 39 High-vs-Max hard-problems comparison (optional, Kaggle)"
for mode in high max; do
  cargo run --quiet --locked --release -p aarambh-studio -- eval \
    --config "$PHASE39_CONFIG" \
    --model "$PHASE39_MODEL" \
    --tokenizer "$PHASE39_TOKENIZER" \
    --tasks hard-problems \
    --data-dir data/eval \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --thinking "$mode" \
    "${EXTRA[@]}" \
    --out "$STATE/scorecard-${mode}.json" \
    --markdown "$STATE/scorecard-${mode}.md"
done

python3 - "$STATE" <<'PY'
import json, sys
from pathlib import Path
state = Path(sys.argv[1])
def find(card):
    for t in card["tasks"]:
        if t["name"] == "hard-problems":
            return t
    return {}
high = find(json.loads((state / "scorecard-high.json").read_text()))
mx = find(json.loads((state / "scorecard-max.json").read_text()))
hd, md = high.get("details", {}), mx.get("details", {})
print("Phase 39 High-vs-Max comparison (hard-problems):")
print(f"  high: accuracy={high.get('value', 0):.4f} thinking={hd.get('thinking_tokens', 0):.2f} completion={hd.get('completion_tokens', 0):.2f} total={hd.get('total_tokens', 0):.2f}")
print(f"  max:  accuracy={mx.get('value', 0):.4f} thinking={md.get('thinking_tokens', 0):.2f} completion={md.get('completion_tokens', 0):.2f} total={md.get('total_tokens', 0):.2f}")
PY

echo "Phase 39 Kaggle comparison helper completed in $STATE"
echo "(Optional script — not run by CI.)"
