#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-data/tool_sft/xlam_train.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-0}"
SEED="${SEED:-42}"

mkdir -p "$(dirname "$OUT")"

python3 - "$OUT" "$MAX_EXAMPLES" "$SEED" <<'PY'
import json
import random
import sys

out_path, max_examples, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
try:
    from datasets import load_dataset
except Exception as exc:
    raise SystemExit(
        "The 'datasets' package is required. Install it with "
        "'python3 -m pip install datasets'.\n" + str(exc)
    )

try:
    rows = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
except Exception as exc:
    raise SystemExit(
        "xLAM is gated. Accept its Hugging Face terms and run 'huggingface-cli login' first.\n"
        + str(exc)
    )

TYPE_MAP = {"dict": "object", "str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array"}

def parsed(value):
    if isinstance(value, str):
        return json.loads(value)
    return value

def normalize_schema(value):
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: normalize_schema(item) for key, item in value.items()}
    if isinstance(result.get("type"), str):
        result["type"] = TYPE_MAP.get(result["type"], result["type"])
    return result

def normalize_tool(tool):
    tool = parsed(tool)
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        tool = tool["function"]
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": normalize_schema(tool.get("parameters", {"type": "object", "properties": {}})),
    }

normalized = []
rejected = 0
for row in rows:
    try:
        tools = [normalize_tool(tool) for tool in parsed(row["tools"])]
        answers = parsed(row["answers"])
        if not isinstance(answers, list) or len(answers) != 1:
            rejected += 1
            continue
        answer = answers[0]
        call = {
            "name": answer["name"],
            "arguments": parsed(answer.get("arguments", {})),
        }
        normalized.append({
            "instruction": row["query"],
            "tools": tools,
            "tool_call": call,
            "response": "",
        })
    except Exception:
        rejected += 1

random.Random(seed).shuffle(normalized)
if max_examples > 0:
    normalized = normalized[:max_examples]
with open(out_path, "w", encoding="utf-8") as handle:
    for row in normalized:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
with open(out_path + ".meta.json", "w", encoding="utf-8") as handle:
    json.dump({
        "source": "Salesforce/xlam-function-calling-60k",
        "seed": seed,
        "written": len(normalized),
        "rejected": rejected,
        "filter": "single-call-only",
    }, handle, indent=2)
print(f"wrote {len(normalized)} examples to {out_path}; rejected {rejected}")
PY
