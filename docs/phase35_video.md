# Phase 35 Native Video Understanding

Phase 35 extends the existing image VLM path to visual video understanding.
It samples frames from H.264 MP4 files, encodes them with the frozen CLIP-style
vision encoder, adds temporal positions, projects the patches into the language
model width, and trains answer-only targets through the shared VLM DoRA trainer.

This is a source implementation. The repository does not ship pretrained video
weights or claim useful NExT-QA accuracy from the two-step smoke fixture.

## Runtime Boundary

- Input container and codec: MP4 with an H.264 visual track.
- Decoder: bundled OpenH264, with MP4 parsing in-process.
- Build requirement: a working C/C++ toolchain for bundled OpenH264 sources.
- Runtime FFmpeg dependency: none.
- Audio: ignored and unsupported as model input.
- Unsupported in Phase 35: other codecs/containers, server upload, video
  self-learning, and video speculative/tool generation.

The fixture generator uses an installed `ffmpeg` command only to create four
small deterministic test clips. Model training, inference, and evaluation do
not invoke it.

## One-Time Vocabulary Migration

Phase 35 reserves `<video>`, `<video_end>`, and `<frame_sep>` as IDs 9, 10, and
11. Existing learned token IDs at 9 and above shift by three without changing
their text or merge rank. SafeTensors embedding rows are expanded at the same
boundary; new rows clone compatible image marker rows.

```sh
target/release/aarambh-ai convert \
  --config configs/vision_vqa_smoke.toml \
  --input checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --output checkpoints/video_smoke/model.safetensors \
  --tokenizer checkpoints/vision_projector_smoke/tokenizer.json \
  --output-tokenizer checkpoints/video_smoke/tokenizer.json \
  --upgrade-video-vocab
```

Migration accepts SafeTensors only. Migrate first and quantize the migrated
checkpoint afterward when GGUF output is needed. Use the migrated model and
tokenizer together; mixing old and new vocabulary layouts is invalid.

## Data

Normalized JSONL records use one example per line:

```json
{"video":"clip.mp4","question":"What happens at the end?","answer":"The light turns green."}
```

`video_path` is accepted as an alias for `video`, and optional `thinking` text
is supported. Relative paths resolve below `[vision.video].video_root`.

Official NExT-QA CSV is accepted directly when it contains `video`, `question`,
`answer`, and `a0` through `a4`. Numeric video IDs receive an `.mp4` suffix,
the options are rendered as A-E, and the target is normalized to one letter.

## Configuration

```toml
[vision.video]
video_root = "data/nextqa/videos"
frame_count = 8
max_frame_count = 8
sampling = "uniform"          # uniform | scene_aware
scene_min_gap = 8
temporal_encoding = "learned" # learned | sinusoidal
temporal_path = "adapters/video_vqa/temporal.safetensors"
encoder_frame_batch_size = 8
feature_cache_entries = 16
```

Uniform sampling includes the first and last decoded frame. Scene-aware
sampling ranks thumbnail luma changes, enforces `scene_min_gap`, and fills the
remaining budget uniformly. Both always return exactly `frame_count` frames,
repeating endpoints for clips shorter than the requested budget.

Lower `frame_count` first when context or activation memory is tight. Lower
`encoder_frame_batch_size` to reduce peak frozen-encoder memory. The bounded
cache stores only detached pre-projector features, so projector and temporal
gradients remain fresh.

## Training

```sh
target/release/aarambh-ai finetune video-dora \
  --config configs/video_qa_smoke.toml \
  --base checkpoints/video_smoke/model.safetensors \
  --tokenizer checkpoints/video_smoke/tokenizer.json \
  --data data/video_smoke/video_qa_smoke_4.jsonl \
  --output adapters/video_qa_smoke \
  --lora-rank 4 \
  --max-steps 2
```

Use `video-qdora` for a quantized base. Training saves adapter metadata,
`adapter.safetensors`, `projector.safetensors`, and, for learned temporal
encoding, `temporal.safetensors`. Image and video examples use the same trainer,
answer mask, optimizer schedule, clipping, and artifact format.

Merge the adapter when ordinary inference should load one model file:

```sh
target/release/aarambh-ai finetune merge \
  --config configs/video_qa_smoke.toml \
  --base checkpoints/video_smoke/model.safetensors \
  --adapter adapters/video_qa_smoke \
  --method dora \
  --output checkpoints/video_qa_smoke_merged
```

The projector and temporal artifacts remain separate and must be referenced by
the inference config.

## Inference And Evaluation

```sh
target/release/aarambh-ai infer \
  --config configs/video_qa_smoke_infer.toml \
  --model checkpoints/video_qa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/video_smoke/tokenizer.json \
  --video data/video_smoke/videos/red_to_blue.mp4 \
  --prompt "What color is shown at the end?" \
  --frames 2 \
  --frame-sampling uniform \
  --max-tokens 8 \
  --greedy \
  --safety none
```

Streaming uses the existing token-by-token safety path after video prefill.
`--frames` and `--frame-sampling` are valid only with `--video`.

```sh
target/release/aarambh-ai eval \
  --config configs/video_qa_smoke_infer.toml \
  --model checkpoints/video_qa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/video_smoke/tokenizer.json \
  --tasks video-qa \
  --data-dir data/eval \
  --max-examples 128 \
  --max-new-tokens 8 \
  --out artifacts/video_qa.json
```

The task aliases `video-qa`, `video_qa`, and `nextqa` report normalized exact
match. `video-qa-smoke` selects the generated fixture.

## Complete Local Smoke

Build the release binary and ensure the Tiny Shakespeare step-50 checkpoint is
available, then run:

```sh
cargo build --release --locked -p aarambh-ai
scripts/phase35_smoke.sh
```

The script generates clips, migrates vocabulary when needed, runs two optimizer
steps, merges, performs native video inference, and writes a scorecard. Its
purpose is execution proof, not model-quality evidence.
