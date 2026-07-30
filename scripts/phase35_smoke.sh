#!/usr/bin/env bash
set -euo pipefail

BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
BASE=${VIDEO_BASE_MODEL:-checkpoints/tiny_shakespeare/step_000050/model.safetensors}
IMAGE_TOKENIZER=${VIDEO_IMAGE_TOKENIZER:-checkpoints/vision_projector_smoke/tokenizer.json}
VIDEO_DIR=${VIDEO_CHECKPOINT_DIR:-checkpoints/video_smoke}
ADAPTER_DIR=${VIDEO_ADAPTER_DIR:-adapters/video_qa_smoke}
MERGED_DIR=${VIDEO_MERGED_DIR:-checkpoints/video_qa_smoke_merged}
SCORECARD=${VIDEO_SCORECARD:-artifacts/phase35_video_smoke.json}

[[ -x "$BIN" ]] || {
  echo "missing executable Phase 35 binary: $BIN" >&2
  exit 2
}
[[ -s "$BASE" ]] || {
  echo "missing Phase 35 base checkpoint: $BASE" >&2
  exit 2
}
command -v ffmpeg >/dev/null || {
  echo "ffmpeg is required only to generate the Phase 35 smoke clips" >&2
  exit 2
}

python3 scripts/phase35_make_video_smoke_fixture.py
[[ -s "$IMAGE_TOKENIZER" ]] || {
  echo "missing generated image tokenizer: $IMAGE_TOKENIZER" >&2
  exit 2
}

mkdir -p "$VIDEO_DIR" "$ADAPTER_DIR" "$MERGED_DIR" "$(dirname "$SCORECARD")"

if [[ ! -s "$VIDEO_DIR/model.safetensors" || ! -s "$VIDEO_DIR/tokenizer.json" ]]; then
  "$BIN" convert \
    --config configs/vision_vqa_smoke.toml \
    --input "$BASE" \
    --output "$VIDEO_DIR/model.safetensors" \
    --tokenizer "$IMAGE_TOKENIZER" \
    --output-tokenizer "$VIDEO_DIR/tokenizer.json" \
    --upgrade-video-vocab
fi

"$BIN" finetune video-dora \
  --config configs/video_qa_smoke.toml \
  --base "$VIDEO_DIR/model.safetensors" \
  --tokenizer "$VIDEO_DIR/tokenizer.json" \
  --data data/video_smoke/video_qa_smoke_4.jsonl \
  --output "$ADAPTER_DIR" \
  --lora-rank 4 \
  --batch-size 1 \
  --max-steps 2 \
  --log-every-n-steps 1 \
  --save-every-n-steps 0

for artifact in adapter.safetensors projector.safetensors temporal.safetensors adapter_config.json; do
  [[ -s "$ADAPTER_DIR/$artifact" ]] || {
    echo "missing Phase 35 training artifact: $ADAPTER_DIR/$artifact" >&2
    exit 2
  }
done

"$BIN" finetune merge \
  --config configs/video_qa_smoke.toml \
  --base "$VIDEO_DIR/model.safetensors" \
  --adapter "$ADAPTER_DIR" \
  --method dora \
  --output "$MERGED_DIR"

"$BIN" infer \
  --config configs/video_qa_smoke_infer.toml \
  --model "$MERGED_DIR/model.safetensors" \
  --tokenizer "$VIDEO_DIR/tokenizer.json" \
  --video data/video_smoke/videos/red_to_blue.mp4 \
  --prompt "What color is shown at the end?" \
  --frames 2 \
  --frame-sampling uniform \
  --max-tokens 2 \
  --greedy \
  --safety none

"$BIN" eval \
  --config configs/video_qa_smoke_infer.toml \
  --model "$MERGED_DIR/model.safetensors" \
  --tokenizer "$VIDEO_DIR/tokenizer.json" \
  --tasks video-qa-smoke \
  --data-dir data/eval \
  --max-examples 2 \
  --max-new-tokens 2 \
  --out "$SCORECARD"

[[ -s "$SCORECARD" ]] || {
  echo "missing Phase 35 scorecard: $SCORECARD" >&2
  exit 2
}
echo "Phase 35 video smoke completed: $SCORECARD"
