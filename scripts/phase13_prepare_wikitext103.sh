#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data}"
ARCHIVE="$DATA_DIR/wikitext-103-raw-v1.zip"
TARGET_DIR="$DATA_DIR/wikitext-103-raw"
URL="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

mkdir -p "$DATA_DIR"

if [[ ! -f "$TARGET_DIR/wiki.train.raw" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    if command -v curl >/dev/null 2>&1; then
      curl -L "$URL" -o "$ARCHIVE"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$ARCHIVE" "$URL"
    else
      echo "curl or wget is required to download WikiText-103" >&2
      exit 1
    fi
  fi
  python3 - <<'PY' "$ARCHIVE" "$DATA_DIR"
import sys
import zipfile

archive, data_dir = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(archive) as zf:
    zf.extractall(data_dir)
PY
fi

test -f "$TARGET_DIR/wiki.train.raw"
echo "$TARGET_DIR/wiki.train.raw"
