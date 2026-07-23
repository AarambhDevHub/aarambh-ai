#!/usr/bin/env bash
set -euo pipefail

BIN=${AARAMBH_BIN:-target/release/aarambh-ai}
BASE=${DOCUMENT_BASE_MODEL:-checkpoints/tiny_shakespeare/step_000050/model.safetensors}
IMAGE_TOKENIZER=${DOCUMENT_IMAGE_TOKENIZER:-checkpoints/vision_projector_smoke/tokenizer.json}
VIDEO_DIR=${DOCUMENT_VIDEO_MIGRATION_DIR:-checkpoints/document_smoke/video}
DOCUMENT_DIR=${DOCUMENT_CHECKPOINT_DIR:-checkpoints/document_smoke}
ADAPTER_DIR=${DOCUMENT_ADAPTER_DIR:-adapters/document_qa_smoke}
MERGED_DIR=${DOCUMENT_MERGED_DIR:-checkpoints/document_qa_smoke_merged}
SCORECARD=${DOCUMENT_SCORECARD:-artifacts/phase36_document_smoke.json}

[[ -x "$BIN" ]] || {
  echo "missing executable Phase 36 binary: $BIN" >&2
  exit 2
}
[[ -s "$BASE" ]] || {
  echo "missing Phase 36 base checkpoint: $BASE" >&2
  exit 2
}

python3 scripts/phase36_make_document_smoke_fixture.py
[[ -s "$IMAGE_TOKENIZER" ]] || {
  echo "missing generated image tokenizer: $IMAGE_TOKENIZER" >&2
  exit 2
}
mkdir -p "$VIDEO_DIR" "$DOCUMENT_DIR" "$ADAPTER_DIR" "$MERGED_DIR" "$(dirname "$SCORECARD")"

if [[ ! -s "$VIDEO_DIR/model.safetensors" || ! -s "$VIDEO_DIR/tokenizer.json" ]]; then
  "$BIN" convert \
    --config configs/vision_vqa_smoke.toml \
    --input "$BASE" \
    --output "$VIDEO_DIR/model.safetensors" \
    --tokenizer "$IMAGE_TOKENIZER" \
    --output-tokenizer "$VIDEO_DIR/tokenizer.json" \
    --upgrade-video-vocab
fi

if [[ ! -s "$DOCUMENT_DIR/model.safetensors" || ! -s "$DOCUMENT_DIR/tokenizer.json" ]]; then
  "$BIN" convert \
    --config configs/video_qa_smoke.toml \
    --input "$VIDEO_DIR/model.safetensors" \
    --output "$DOCUMENT_DIR/model.safetensors" \
    --tokenizer "$VIDEO_DIR/tokenizer.json" \
    --output-tokenizer "$DOCUMENT_DIR/tokenizer.json" \
    --upgrade-document-vocab
fi

"$BIN" finetune document-dora \
  --config configs/document_qa_smoke.toml \
  --base "$DOCUMENT_DIR/model.safetensors" \
  --tokenizer "$DOCUMENT_DIR/tokenizer.json" \
  --data data/document_smoke/document_qa_smoke_4.jsonl \
  --output "$ADAPTER_DIR" \
  --lora-rank 4 \
  --batch-size 1 \
  --max-steps 2 \
  --log-every-n-steps 1 \
  --save-every-n-steps 0

for artifact in adapter.safetensors projector.safetensors layout.safetensors adapter_config.json; do
  [[ -s "$ADAPTER_DIR/$artifact" ]] || {
    echo "missing Phase 36 training artifact: $ADAPTER_DIR/$artifact" >&2
    exit 2
  }
done

"$BIN" finetune merge \
  --config configs/document_qa_smoke.toml \
  --base "$DOCUMENT_DIR/model.safetensors" \
  --adapter "$ADAPTER_DIR" \
  --method dora \
  --output "$MERGED_DIR"

"$BIN" infer \
  --config configs/document_qa_smoke_infer.toml \
  --model "$MERGED_DIR/model.safetensors" \
  --tokenizer "$DOCUMENT_DIR/tokenizer.json" \
  --document data/document_smoke/documents/red_invoice.pdf \
  --prompt "What color fills the first page?" \
  --max-tokens 2 \
  --greedy \
  --safety none

"$BIN" eval \
  --config configs/document_qa_smoke_infer.toml \
  --model "$MERGED_DIR/model.safetensors" \
  --tokenizer "$DOCUMENT_DIR/tokenizer.json" \
  --tasks document-qa-smoke \
  --data-dir data/eval \
  --max-examples 2 \
  --max-new-tokens 2 \
  --out "$SCORECARD"

[[ -s "$SCORECARD" ]] || {
  echo "missing Phase 36 scorecard: $SCORECARD" >&2
  exit 2
}
echo "Phase 36 document smoke completed: $SCORECARD"
