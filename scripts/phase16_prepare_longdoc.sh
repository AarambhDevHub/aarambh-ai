#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
RAW_DIR="$DATA_DIR/wikitext-103-raw"
OUTPUT_DIR="$DATA_DIR/long_context"
OUTPUT="$OUTPUT_DIR/wikitext103_longdoc.txt"

scripts/phase13_prepare_wikitext103.sh "$DATA_DIR" >/dev/null
mkdir -p "$OUTPUT_DIR"

python3 - <<'PY' "$RAW_DIR/wiki.train.raw" "$OUTPUT"
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
paragraphs = []

for line in src.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()
    if not line or (line.startswith("=") and line.endswith("=")):
        continue
    paragraphs.append(line)

if not paragraphs:
    raise SystemExit(f"no usable text found in {src}")

dst.write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
print(dst)
PY
