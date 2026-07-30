#!/usr/bin/env bash
set -euo pipefail

BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
CONFIG=${QAT_CONFIG:-configs/qat_smoke.toml}
CHECKPOINT_DIR=${QAT_CHECKPOINT_DIR:-checkpoints/qat_smoke}

for path in "$BIN" "$CONFIG" data/tiny_shakespeare.txt; do
  [[ -e "$path" ]] || {
    echo "missing Phase 34 smoke input: $path" >&2
    exit 2
  }
done
[[ -x "$BIN" ]] || {
  echo "binary is not executable: $BIN" >&2
  exit 2
}
rg -q '^\[model\.qat\]' "$CONFIG" || {
  echo "QAT smoke config must contain [model.qat]" >&2
  exit 2
}

"$BIN" train --config "$CONFIG"

STATE="$CHECKPOINT_DIR/step_000002/train_state.json"
MODEL="$CHECKPOINT_DIR/step_000002/model.safetensors"
for path in "$STATE" "$MODEL"; do
  [[ -s "$path" ]] || {
    echo "missing QAT smoke output: $path" >&2
    exit 2
  }
done
rg -q '"qat"' "$STATE" || {
  echo "QAT checkpoint does not persist its policy" >&2
  exit 2
}

echo "Phase 34 QAT smoke completed: $CHECKPOINT_DIR"
