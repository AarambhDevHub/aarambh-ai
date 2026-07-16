#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <baseline-config> <mtp-config> <eval-data-dir> <output-dir>" >&2
  exit 2
fi

BASELINE_CONFIG=$1
MTP_CONFIG=$2
DATA_DIR=$3
OUTPUT_DIR=$4
BIN=${AARAMBH_BIN:-target/release/aarambh-ai}
TASKS=${TASKS:-ppl}
MAX_EXAMPLES=${MAX_EXAMPLES:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

for path in "$BASELINE_CONFIG" "$MTP_CONFIG"; do
  [[ -f "$path" ]] || {
    echo "missing config: $path" >&2
    exit 2
  }
done
[[ -d "$DATA_DIR" ]] || {
  echo "missing eval data directory: $DATA_DIR" >&2
  exit 2
}
[[ -x "$BIN" ]] || {
  echo "missing $BIN; build the release binary before comparison" >&2
  exit 2
}
rg -q '^\[model\.mtp\]' "$MTP_CONFIG" || {
  echo "MTP config must contain [model.mtp]" >&2
  exit 2
}
if rg -q '^\[model\.mtp\]' "$BASELINE_CONFIG"; then
  echo "baseline config must not contain [model.mtp]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

run_case() {
  local label=$1
  local config=$2
  local score="$OUTPUT_DIR/$label.json"
  local markdown="$OUTPUT_DIR/$label.md"

  echo "training Phase 32 case: $label"
  "$BIN" train --config "$config" 2>&1 | tee "$OUTPUT_DIR/$label.train.log"
  "$BIN" eval \
    --config "$config" \
    --tasks "$TASKS" \
    --data-dir "$DATA_DIR" \
    --max-examples "$MAX_EXAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "$score" \
    --markdown "$markdown"
}

run_case baseline "$BASELINE_CONFIG"
run_case mtp "$MTP_CONFIG"
"$BIN" eval \
  --compare "$OUTPUT_DIR/baseline.json" "$OUTPUT_DIR/mtp.json" \
  --markdown "$OUTPUT_DIR/comparison.md"

echo "Phase 32 training comparison written to $OUTPUT_DIR/comparison.md"
