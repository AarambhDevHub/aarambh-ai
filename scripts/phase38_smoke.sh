#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("data/eval/forgetting/probes.json").read_text())
schema = json.loads(Path("schemas/manas-forgetting-v1.schema.json").read_text())
assert manifest["schema_version"] == 1
assert len(manifest["probes"]) == 8
assert len({probe["capability"] for probe in manifest["probes"]}) == 8
assert schema["additionalProperties"] is False
assert len(schema["required"]) == 7
print("Phase 38 contracts: valid")
PY

cargo test --locked -p aarambh-ai-eval forgetting
cargo test --locked -p aarambh-ai-selflearn forgetting
cargo run --quiet --locked -p aarambh-ai -- eval --help |
  rg --quiet -- '--forgetting-manifest'
cargo run --quiet --locked -p aarambh-ai -- selflearn --help |
  rg --quiet -- 'forgetting-report'

if [[ -n "${PHASE38_MODEL:-}" ]]; then
  PHASE38_CONFIG="${PHASE38_CONFIG:-configs/tiny_shakespeare.toml}"
  PHASE38_TOKENIZER="${PHASE38_TOKENIZER:-checkpoints/tiny_shakespeare/tokenizer.json}"
  PHASE38_STATE="${PHASE38_STATE:-${TMPDIR:-/tmp}/aarambh-phase38-smoke-$$}"
  mkdir -p "$PHASE38_STATE"

  for checkpoint_id in smoke-baseline smoke-current; do
    cargo run --quiet --locked -p aarambh-ai -- eval \
      --config "$PHASE38_CONFIG" \
      --model "$PHASE38_MODEL" \
      --tokenizer "$PHASE38_TOKENIZER" \
      --tasks gsm8k \
      --data-dir data/eval \
      --max-examples 2 \
      --max-new-tokens 8 \
      --forgetting-manifest data/eval/forgetting/probes.json \
      --forgetting-store "$PHASE38_STATE/curves.json" \
      --checkpoint-id "$checkpoint_id" \
      --baseline-id smoke-baseline \
      --forgetting-jsonl "$PHASE38_STATE/manas.jsonl"
  done

  test -s "$PHASE38_STATE/curves.json"
  test -f "$PHASE38_STATE/manas.jsonl"
  echo "Phase 38 checkpoint smoke completed in $PHASE38_STATE"
else
  echo "PHASE38_MODEL is unset; checkpoint execution smoke was skipped"
fi

echo "Phase 38 source and contract smoke completed"
