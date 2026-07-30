#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <qat-config> <baseline-model> <qat-model> <eval-data-dir> <output-dir>" >&2
  exit 2
fi

CONFIG=$1
BASELINE=$2
QAT_MODEL=$3
DATA_DIR=$4
OUTPUT_DIR=$5
BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
TASKS=${TASKS:-ppl}
MAX_EXAMPLES=${MAX_EXAMPLES:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}

for path in "$BIN" "$CONFIG" "$BASELINE" "$QAT_MODEL"; do
  [[ -e "$path" ]] || {
    echo "missing Phase 34 comparison input: $path" >&2
    exit 2
  }
done
[[ -d "$DATA_DIR" ]] || {
  echo "missing eval data directory: $DATA_DIR" >&2
  exit 2
}
rg -q '^\[model\.qat\]' "$CONFIG" || {
  echo "comparison config must contain [model.qat]" >&2
  exit 2
}

mkdir -p "$OUTPUT_DIR"
"$BIN" eval \
  --config "$CONFIG" \
  --model "$QAT_MODEL" \
  --baseline-model "$BASELINE" \
  --qat-compare \
  --tasks "$TASKS" \
  --data-dir "$DATA_DIR" \
  --max-examples "$MAX_EXAMPLES" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --out "$OUTPUT_DIR/qat_robustness.json" \
  --markdown "$OUTPUT_DIR/qat_robustness.md"

echo "Phase 34 robustness report: $OUTPUT_DIR/qat_robustness.md"
