#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data}"
SPLIT="${COCO_SPLIT:-val2017}"
MAX_IMAGES="${MAX_IMAGES:-5000}"
OUT_DIR="$ROOT/coco_captions"
ANN_DIR="$OUT_DIR/annotations"
IMG_DIR="$OUT_DIR/images"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL="http://images.cocodataset.org/zips/${SPLIT}.zip"

mkdir -p "$ANN_DIR" "$IMG_DIR"

download() {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    return
  fi
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "curl or wget is required" >&2
    exit 1
  fi
}

download "$ANN_URL" "$OUT_DIR/annotations_trainval2017.zip"
download "$IMG_URL" "$OUT_DIR/${SPLIT}.zip"

if [[ ! -f "$ANN_DIR/captions_${SPLIT}.json" ]]; then
  unzip -q "$OUT_DIR/annotations_trainval2017.zip" -d "$OUT_DIR"
fi

if [[ ! -d "$IMG_DIR/$SPLIT" ]]; then
  unzip -q "$OUT_DIR/${SPLIT}.zip" -d "$IMG_DIR"
fi

python3 - "$ANN_DIR/captions_${SPLIT}.json" "$IMG_DIR/$SPLIT" "$OUT_DIR/train.jsonl" "$MAX_IMAGES" <<'PY'
import json
import sys
from pathlib import Path

ann_path = Path(sys.argv[1])
image_root = Path(sys.argv[2])
out_path = Path(sys.argv[3])
max_images = int(sys.argv[4])

data = json.loads(ann_path.read_text())
images = {item["id"]: item["file_name"] for item in data["images"]}
seen = set()
written = 0

with out_path.open("w", encoding="utf-8") as out:
    for item in data["annotations"]:
        image_id = item["image_id"]
        if image_id not in images or image_id in seen:
            continue
        file_name = images[image_id]
        if not (image_root / file_name).exists():
            continue
        seen.add(image_id)
        out.write(json.dumps({
            "image": file_name,
            "caption": item["caption"],
        }, ensure_ascii=False) + "\n")
        written += 1
        if written >= max_images:
            break

print(f"wrote {written} captions to {out_path}")
PY

echo "COCO caption JSONL: $OUT_DIR/train.jsonl"
echo "Image root:         $IMG_DIR/$SPLIT"
