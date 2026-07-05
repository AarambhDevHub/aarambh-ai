#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/eval}"
MAX_EXAMPLES="${2:-128}"

mkdir -p "$OUT_DIR"/{ppl,mmlu_lite,hellaswag,gsm8k_subset,humaneval_lite}

python3 - <<'PY' "$OUT_DIR" "$MAX_EXAMPLES"
import json
import pathlib
import sys

out_dir = pathlib.Path(sys.argv[1])
max_examples = int(sys.argv[2])

try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit(
        "Python package 'datasets' is required for public eval downloads. "
        "Install it with: python3 -m pip install datasets\n"
        f"Import error: {exc}"
    )

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def take(dataset, limit):
    rows = []
    for row in dataset:
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows

# Perplexity holdout: WikiText-103 validation.
wikitext = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"validation[:{max_examples}]")
holdout = "\n".join(row["text"] for row in wikitext if row.get("text", "").strip())
(out_dir / "ppl" / "holdout.txt").write_text(holdout + "\n", encoding="utf-8")

# MMLU-lite. Prefer cais/mmlu, all validation rows.
mmlu = load_dataset("cais/mmlu", "all", split=f"validation[:{max_examples}]")
write_jsonl(
    out_dir / "mmlu_lite" / "data.jsonl",
    (
        {
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        }
        for row in take(mmlu, max_examples)
    ),
)

hellaswag = load_dataset("Rowan/hellaswag", split=f"validation[:{max_examples}]")
write_jsonl(
    out_dir / "hellaswag" / "data.jsonl",
    (
        {
            "context": row["ctx"],
            "endings": row["endings"],
            "label": row["label"],
        }
        for row in take(hellaswag, max_examples)
    ),
)

gsm8k = load_dataset("openai/gsm8k", "main", split=f"test[:{max_examples}]")
write_jsonl(
    out_dir / "gsm8k_subset" / "data.jsonl",
    (
        {
            "question": row["question"],
            "answer": row["answer"],
        }
        for row in take(gsm8k, max_examples)
    ),
)

humaneval = load_dataset("openai/openai_humaneval", split=f"test[:{max_examples}]")
write_jsonl(
    out_dir / "humaneval_lite" / "data.jsonl",
    (
        {
            "prompt": row["prompt"],
            "test": row["test"],
        }
        for row in take(humaneval, max_examples)
    ),
)

print(out_dir)
PY
