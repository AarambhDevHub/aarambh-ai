#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/vision}"
MODEL_URL="${CLIP_MODEL_URL:-https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/model.safetensors}"

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/clip_b32_config.json" <<'JSON'
{
  "patch_size": 32,
  "image_size": 224,
  "in_channels": 3,
  "vit_d_model": 768,
  "vit_layers": 12,
  "vit_heads": 12,
  "mlp_dim": 3072,
  "num_patches": 49,
  "norm_eps": 0.00001
}
JSON

if [[ ! -f "$OUT_DIR/clip_b32.safetensors" ]]; then
  if command -v curl >/dev/null 2>&1; then
    curl -L "$MODEL_URL" -o "$OUT_DIR/clip_b32.safetensors"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$OUT_DIR/clip_b32.safetensors" "$MODEL_URL"
  else
    echo "curl or wget is required to download CLIP weights" >&2
    exit 1
  fi
fi

echo "CLIP-B/32 config:  $OUT_DIR/clip_b32_config.json"
echo "CLIP-B/32 weights: $OUT_DIR/clip_b32.safetensors"
