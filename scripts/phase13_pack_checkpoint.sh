#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${1:?usage: scripts/phase13_pack_checkpoint.sh <checkpoint-dir> [output.zip]}"
OUTPUT="${2:-phase13_checkpoint.zip}"

python3 - <<'PY' "$CHECKPOINT_DIR" "$OUTPUT"
import pathlib
import sys
import zipfile

checkpoint_dir = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
if not checkpoint_dir.exists():
    raise SystemExit(f"checkpoint directory does not exist: {checkpoint_dir}")

patterns = [
    "latest.json",
    "best.json",
    "tokenizer.json",
    "**/model.safetensors",
    "**/optimizer.safetensors",
    "**/state.json",
    "**/*.log",
]

files = []
for pattern in patterns:
    files.extend(path for path in checkpoint_dir.glob(pattern) if path.is_file())

if not files:
    raise SystemExit(f"no checkpoint artifacts found in {checkpoint_dir}")

with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for path in sorted(set(files)):
        zf.write(path, path.relative_to(checkpoint_dir.parent))

print(output)
PY
