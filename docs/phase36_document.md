# Phase 36 Native Document Understanding

Phase 36 extends the existing image/video VLM path to PDFs and ordered scanned
pages. Documents are rendered to RGB pages, passed through the same frozen
CLIP-style encoder, projected into language-model width with 2D patch positions,
and fused at `<document>`/`<page_sep>` marker positions.

## Scope

- PDF rendering: pinned pure-Rust Hayro 0.4 with JPEG2000 support.
- Raster sources: one image or an ordered `page_paths` array.
- Defaults: 150 DPI, at most 16 pages, 32 million pixels per page.
- Layout: learned or sinusoidal row/column offsets after the base projector.
- Training: shared DoRA/QDoRA VLM loop; frozen vision encoder.
- Evaluation: ANLS primary, exact match secondary, optional table ANLS.
- Not included: OCR, table parsing, server document uploads, document
  self-learning, speculative document decoding, or document tool calling.

## Data

Canonical JSONL accepts one document source and one or more answers:

```json
{"document":"invoices/inv_042.pdf","question":"What is the total?","answers":["$4,230.00"],"pages":[2],"tags":["table"]}
{"page_paths":["scan/page_1.png","scan/page_2.png"],"question":"Who signed this form?","answer":"A. Kumar"}
```

Paths are resolved under `vision.document.document_root`. `pages` is optional,
1-based, ordered, and cannot contain duplicates. Without it, the first
`max_pages_per_document` pages are used.

DocVQA and MP-DocVQA require the user to accept their terms and download the
original files. Normalize a downloaded annotation file with:

```sh
python3 scripts/phase36_prepare_docvqa.py \
  --annotations /path/to/annotations.json \
  --documents-dir /path/to/documents \
  --output data/document_qa/train.jsonl
```

## Vocabulary Migration

Phase 36 reserves IDs 12-14. Migrate a video-capable tokenizer and matching
SafeTensors model before training:

```sh
target/release/aarambh-ai convert \
  --config configs/video_qa_smoke.toml \
  --input checkpoints/video_smoke/model.safetensors \
  --output checkpoints/document_smoke/model.safetensors \
  --tokenizer checkpoints/video_smoke/tokenizer.json \
  --output-tokenizer checkpoints/document_smoke/tokenizer.json \
  --upgrade-document-vocab
```

The migration shifts learned IDs at 12 and above by three and expands tied or
untied vocabulary tensors in lockstep. Do not pair a migrated model with the
old tokenizer.

## Training

```sh
target/release/aarambh-ai finetune document-dora \
  --config configs/document_qa_smoke.toml \
  --base checkpoints/document_smoke/model.safetensors \
  --tokenizer checkpoints/document_smoke/tokenizer.json \
  --data data/document_smoke/document_qa_smoke_4.jsonl \
  --output adapters/document_qa_smoke \
  --lora-rank 4 \
  --batch-size 1 \
  --max-steps 2
```

Use `document-qdora` for a quantized base. Learned layout runs save
`layout.safetensors`; point `vision.document.layout_path` at it for inference
and evaluation. `projector.safetensors` is saved alongside it.

The frozen encoder runs pages in chunks controlled by
`encoder_page_batch_size`. `feature_cache_entries` bounds detached
pre-projector feature caching; set it to zero to disable caching.

## Inference

```sh
target/release/aarambh-ai infer \
  --config configs/document_qa_smoke_infer.toml \
  --model checkpoints/document_qa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/document_smoke/tokenizer.json \
  --document data/document_smoke/documents/red_invoice.pdf \
  --pages 1,2 \
  --prompt "What color fills the first page?" \
  --max-tokens 8 \
  --greedy \
  --safety strict \
  --stream
```

`--document-dpi` and `--max-document-pages` override configured rasterization
limits for one request. Safety scans generated stream tokens exactly as it does
after image/video multimodal prefill.

## Evaluation

```sh
target/release/aarambh-ai eval \
  --config configs/document_qa_smoke_infer.toml \
  --model checkpoints/document_qa_smoke_merged/model.safetensors \
  --tokenizer checkpoints/document_smoke/tokenizer.json \
  --tasks document-qa \
  --data-dir data/eval \
  --max-new-tokens 32 \
  --out artifacts/document_qa.json
```

ANLS takes the best normalized Levenshtein similarity across accepted answers
and zeros similarities below 0.5. Exact match is reported as a secondary
metric. Records tagged `table` also produce `table_anls`.

## Local Smoke

```sh
cargo build --release -p aarambh-ai
python3 scripts/phase36_make_document_smoke_fixture.py
scripts/phase36_smoke.sh
```

The fixture creates four two-page PDFs without an external PDF tool. The smoke
run performs image-to-video and video-to-document vocabulary migrations,
two DoRA steps, adapter merge, PDF inference, and two-example ANLS evaluation.
It proves execution, not useful document-answering quality.
