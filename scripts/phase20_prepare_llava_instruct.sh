#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-data}"
OUT_DIR="$OUT_ROOT/llava"
JSON_URL="${LLAVA_INSTRUCT_URL:-https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"

mkdir -p "$OUT_DIR"

RAW_JSON="$OUT_DIR/llava_instruct_150k.json"
JSONL="$OUT_DIR/llava_instruct_150k.jsonl"

if [[ ! -f "$RAW_JSON" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "$JSON_URL" -o "$RAW_JSON"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$RAW_JSON" "$JSON_URL"
  else
    echo "curl or wget is required to download LLaVA-Instruct-150K metadata" >&2
    exit 1
  fi
fi

python3 - "$RAW_JSON" "$JSONL" "$MAX_EXAMPLES" <<'PY'
import json
import sys

src, dst, max_examples = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, "r", encoding="utf-8") as f:
    data = json.load(f)

written = 0
with open(dst, "w", encoding="utf-8") as out:
    for row in data:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
        if max_examples > 0 and written >= max_examples:
            break

print(f"wrote {written} examples to {dst}")
PY

echo "LLaVA metadata JSONL: $JSONL"
echo "Images are not bundled with LLaVA-Instruct-150K."
echo "Place matching COCO/train2017 images under: $OUT_ROOT/llava/images"
