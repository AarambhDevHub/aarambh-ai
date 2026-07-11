#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-data/eval/tool_calling/data.jsonl}"
MAX_EXAMPLES="${MAX_EXAMPLES:-128}"
BFCL_REF="${BFCL_REF:-v1.2}"
SOURCE_DIR="${BFCL_SOURCE_DIR:-}"
TMP=""

if [[ -z "$SOURCE_DIR" ]]; then
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  git clone --quiet --depth 1 --branch "$BFCL_REF" https://github.com/ShishirPatil/gorilla.git "$TMP/gorilla"
  SOURCE_DIR="$TMP/gorilla/berkeley-function-call-leaderboard"
fi

mkdir -p "$(dirname "$OUT")"

python3 - "$SOURCE_DIR" "$OUT" "$MAX_EXAMPLES" "$BFCL_REF" <<'PY'
import ast
import json
import pathlib
import sys

root, out_path, limit, source_ref = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]

def locate(name, answers=False):
    matches = [path for path in root.rglob(name) if ("possible_answer" in str(path)) == answers]
    if not matches:
        raise SystemExit(f"unable to locate {'answers for ' if answers else ''}{name} under {root}")
    return matches[0]

def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return value["content"]
        return "\n".join(filter(None, (text(item) for item in value.values())))
    if isinstance(value, list):
        return "\n".join(filter(None, (text(item) for item in value)))
    return ""

def literal(node):
    return ast.literal_eval(node)

def parse_call(value):
    if isinstance(value, dict) and "name" in value:
        return {"name": value["name"], "arguments": value.get("arguments", {})}
    if not isinstance(value, str):
        return None
    node = ast.parse(value.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        return None
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        name = ".".join(reversed(parts))
    else:
        return None
    return {"name": name, "arguments": {item.arg: literal(item.value) for item in node.keywords if item.arg}}

rows = []
for category in ("simple", "multiple", "irrelevance"):
    name = f"BFCL_v3_{category}.json"
    cases = read_jsonl(locate(name))
    answers = {row.get("id"): row for row in read_jsonl(locate(name, answers=True))}
    for case in cases:
        tools = case.get("function", case.get("tools", []))
        instruction = text(case.get("question", case.get("messages", ""))).strip()
        if not instruction or not isinstance(tools, list):
            continue
        expected = None
        if category != "irrelevance":
            answer = answers.get(case.get("id"), {})
            candidates = answer.get("ground_truth", answer.get("possible_answer", answer.get("answer", [])))
            if not isinstance(candidates, list):
                candidates = [candidates]
            expected = next((call for call in map(parse_call, candidates) if call), None)
            if expected is None:
                continue
        rows.append({"instruction": instruction, "tools": tools, "tool_call": expected})
        if len(rows) >= limit:
            break
    if len(rows) >= limit:
        break

with out_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
with (out_path.with_suffix(out_path.suffix + ".meta.json")).open("w", encoding="utf-8") as handle:
    json.dump({"source": "ShishirPatil/gorilla", "ref": source_ref, "written": len(rows)}, handle, indent=2)
print(f"wrote {len(rows)} BFCL examples to {out_path}")
PY
