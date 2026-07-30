#!/usr/bin/env bash
# Phase 39 — Max thinking mode smoke test.
#
# Verifies that the new `max` thinking mode is wired through every relevant
# command and that the High-vs-Max comparison path runs end to end when a
# model checkpoint is supplied. Source-level and contract checks always run;
# checkpoint execution is gated on PHASE39_MODEL (the same opt-in pattern used
# by phases 27–38).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Phase 39 unit + contract tests"
cargo test --locked -p aarambh-studio-inference thinking
cargo test --locked -p aarambh-studio-finetune --lib grpo
cargo test --locked -p aarambh-studio-distill config
cargo test --locked -p aarambh-studio-eval --lib hard_problems
cargo test --locked -p aarambh-studio-eval --lib generation

echo "==> Phase 39 CLI surface accepts --thinking max"
cargo run --quiet --locked -p aarambh-studio -- infer --help |
  rg --quiet -- 'none, low, medium, high, or max'
cargo run --quiet --locked -p aarambh-studio -- agent --help |
  rg --quiet -- 'none, low, medium, high, or max'
cargo run --quiet --locked -p aarambh-studio -- eval --help |
  rg --quiet -- 'none, low, medium, high, or max'
cargo run --quiet --locked -p aarambh-studio -- serve --help |
  rg --quiet -- 'thinking'
cargo run --quiet --locked -p aarambh-studio -- finetune grpo --help |
  rg --quiet -- 'thinking'
cargo run --quiet --locked -p aarambh-studio -- distill train --help |
  rg --quiet -- 'thinking'
cargo run --quiet --locked -p aarambh-studio -- selflearn start --help |
  rg --quiet -- 'none, low, medium, high, or max'

echo "==> Phase 39 hard-problems fixture is well-formed"
python3 - <<'PY'
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("data/eval/hard_problems/data.jsonl").read_text().splitlines() if line.strip()]
assert len(rows) >= 4, "hard_problems fixture should contain several deterministic problems"
for row in rows:
    assert "question" in row and "answer" in row, row
    assert row["answer"].startswith("#### "), row
print(f"Phase 39 fixture: {len(rows)} deterministic hard problems")
PY

if [[ -n "${PHASE39_MODEL:-}" ]]; then
  PHASE39_CONFIG="${PHASE39_CONFIG:-configs/tiny_shakespeare.toml}"
  PHASE39_TOKENIZER="${PHASE39_TOKENIZER:-checkpoints/tiny_shakespeare/tokenizer.json}"
  PHASE39_STATE="${PHASE39_STATE:-${TMPDIR:-/tmp}/aarambh-phase39-smoke-$$}"
  mkdir -p "$PHASE39_STATE"

  echo "==> Phase 39 infer smoke (max mode)"
  cargo run --quiet --locked -p aarambh-studio -- infer \
    --config "$PHASE39_CONFIG" \
    --model "$PHASE39_MODEL" \
    --tokenizer "$PHASE39_TOKENIZER" \
    --prompt "What is 17 times 23?" \
    --max-tokens 64 \
    --thinking max

  echo "==> Phase 39 eval High-vs-Max comparison on hard-problems"
  for mode in high max; do
    cargo run --quiet --locked -p aarambh-studio -- eval \
      --config "$PHASE39_CONFIG" \
      --model "$PHASE39_MODEL" \
      --tokenizer "$PHASE39_TOKENIZER" \
      --tasks hard-problems \
      --data-dir data/eval \
      --max-examples 2 \
      --max-new-tokens 96 \
      --thinking "$mode" \
      --out "$PHASE39_STATE/scorecard-${mode}.json"
  done

  python3 - <<PY
import json
from pathlib import Path
state = Path("$PHASE39_STATE")
high = json.loads((state / "scorecard-high.json").read_text())
mx = json.loads((state / "scorecard-max.json").read_text())
def details(card):
    for t in card["tasks"]:
        if t["name"] == "hard-problems":
            return t.get("details", {})
    return {}
hd, md = details(high), details(mx)
print("Phase 39 High-vs-Max comparison:")
print(f"  high: thinking_tokens={hd.get('thinking_tokens', 0):.2f} completion_tokens={hd.get('completion_tokens', 0):.2f} total_tokens={hd.get('total_tokens', 0):.2f}")
print(f"  max:  thinking_tokens={md.get('thinking_tokens', 0):.2f} completion_tokens={md.get('completion_tokens', 0):.2f} total_tokens={md.get('total_tokens', 0):.2f}")
assert md.get('total_tokens', 0) >= hd.get('total_tokens', 0) - 1e-6, "max should spend at least as many tokens as high"
PY
  echo "Phase 39 checkpoint smoke completed in $PHASE39_STATE"
else
  echo "PHASE39_MODEL is unset; checkpoint execution smoke was skipped"
fi

echo "Phase 39 smoke completed"
