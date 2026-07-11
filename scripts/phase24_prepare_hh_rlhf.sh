#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/dpo/hh_rlhf}"
HH_SUBSET="${HH_SUBSET:-helpful-base}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR/eval/preference"

python3 - "$OUT_DIR" "$HH_SUBSET" "$MAX_EXAMPLES" "$SEED" <<'PY'
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
subset = sys.argv[2]
max_examples = int(sys.argv[3])
seed = int(sys.argv[4])

try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit(
        "Python package 'datasets' is required. Install it with: "
        "python3 -m pip install datasets\n"
        f"Import error: {exc}"
    )

marker = "\n\nAssistant:"

def normalize(row):
    chosen = row.get("chosen", "")
    rejected = row.get("rejected", "")
    chosen_at = chosen.rfind(marker)
    rejected_at = rejected.rfind(marker)
    if chosen_at < 0 or rejected_at < 0:
        return None
    chosen_prompt = chosen[:chosen_at].strip()
    rejected_prompt = rejected[:rejected_at].strip()
    if chosen_prompt != rejected_prompt:
        return None
    chosen_response = chosen[chosen_at + len(marker):].strip()
    rejected_response = rejected[rejected_at + len(marker):].strip()
    if not chosen_prompt or not chosen_response or not rejected_response:
        return None
    if chosen_response == rejected_response:
        return None
    return {
        "prompt": chosen_prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
    }

def load(split):
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    dataset = dataset.shuffle(seed=seed)
    rows = []
    for row in dataset:
        normalized = normalize(row)
        if normalized is not None:
            rows.append(normalized)
        if max_examples > 0 and len(rows) >= max_examples:
            break
    return rows

def write(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

train = load("train")
test = load("test")
if not train or not test:
    raise SystemExit("HH-RLHF normalization produced an empty train or test split")
write(out_dir / "train.jsonl", train)
write(out_dir / "eval" / "preference" / "data.jsonl", test)
print(f"wrote {len(train)} train pairs and {len(test)} eval pairs to {out_dir}")
PY

echo "WARNING: HH-RLHF can contain offensive or upsetting material."
echo "DPO train: $OUT_DIR/train.jsonl"
echo "DPO eval:  $OUT_DIR/eval/preference/data.jsonl"
