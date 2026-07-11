#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/dpo/ultrafeedback}"
DATASET="${ULTRAFEEDBACK_DATASET:-trl-lib/ultrafeedback_binarized}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
EVAL_FRACTION="${EVAL_FRACTION:-0.05}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR/eval/preference"

python3 - "$OUT_DIR" "$DATASET" "$MAX_EXAMPLES" "$EVAL_FRACTION" "$SEED" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
dataset_name = sys.argv[2]
max_examples = int(sys.argv[3])
eval_fraction = float(sys.argv[4])
seed = int(sys.argv[5])

if not 0.0 < eval_fraction < 1.0:
    raise SystemExit("EVAL_FRACTION must be between 0 and 1")

try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit(
        "Python package 'datasets' is required. Install it with: "
        "python3 -m pip install datasets\n"
        f"Import error: {exc}"
    )

def message_text(value, response=False):
    if isinstance(value, str):
        return value.strip()
    if not isinstance(value, list):
        return ""
    messages = [item for item in value if isinstance(item, dict)]
    if response:
        assistants = [
            str(item.get("content", "")).strip()
            for item in messages
            if item.get("role") == "assistant" and str(item.get("content", "")).strip()
        ]
        return assistants[-1] if assistants else ""
    rendered = []
    for item in messages:
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        role = str(item.get("role", "user")).strip().capitalize()
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)

def normalize(row):
    prompt = message_text(row.get("prompt", row.get("instruction", "")))
    chosen = message_text(row.get("chosen", ""), response=True)
    rejected = message_text(row.get("rejected", ""), response=True)
    if not prompt or not chosen or not rejected or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

dataset = load_dataset(dataset_name, split="train").shuffle(seed=seed)
rows = []
for row in dataset:
    normalized = normalize(row)
    if normalized is not None:
        rows.append(normalized)
    if max_examples > 0 and len(rows) >= max_examples:
        break

if len(rows) < 2:
    raise SystemExit("UltraFeedback normalization produced fewer than two pairs")
eval_count = max(1, round(len(rows) * eval_fraction))
eval_rows = rows[:eval_count]
train_rows = rows[eval_count:]
if not train_rows:
    raise SystemExit("EVAL_FRACTION left no training pairs")

def write(path, values):
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

write(out_dir / "train.jsonl", train_rows)
write(out_dir / "eval" / "preference" / "data.jsonl", eval_rows)
print(f"wrote {len(train_rows)} train pairs and {len(eval_rows)} eval pairs to {out_dir}")
PY

echo "DPO train: $OUT_DIR/train.jsonl"
echo "DPO eval:  $OUT_DIR/eval/preference/data.jsonl"
