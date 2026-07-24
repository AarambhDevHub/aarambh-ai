#!/usr/bin/env python3
"""Normalize explicit BFCL v1.3 multi-turn response paths for Phase 37 eval.

This importer intentionally keeps only records that already contain ordered
assistant calls, caller-visible text results, and a final answer. It does not
execute BFCL environments or claim full BFCL state-machine coverage.
"""

import argparse
import json
from pathlib import Path


def lines(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tool_definition(raw):
    if raw.get("type") == "function":
        raw = raw["function"]
    return {
        "name": raw["name"],
        "description": raw.get("description", ""),
        "parameters": raw.get(
            "parameters", {"type": "object", "properties": {}}
        ),
    }


def call(raw):
    if "function" in raw and isinstance(raw["function"], dict):
        raw = raw["function"]
    arguments = raw.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return {"name": raw["name"], "arguments": arguments}


def normalize(record):
    tools = record.get("tools", record.get("function", []))
    messages = record.get("messages", record.get("question", []))
    if not isinstance(tools, list) or not isinstance(messages, list):
        return None
    instruction = next(
        (
            item.get("content", "")
            for item in messages
            if item.get("role") == "user" and isinstance(item.get("content"), str)
        ),
        "",
    ).strip()
    steps = []
    final_answer = ""
    pending = []
    for message in messages:
        role = message.get("role")
        calls = message.get("tool_calls", [])
        if role == "assistant" and calls:
            pending.extend(call(item) for item in calls)
        elif role in ("tool", "function") and pending:
            expected = pending.pop(0)
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            steps.append(
                {
                    "call": expected,
                    "result": {
                        "call_id": f"call_{len(steps) + 1:04d}",
                        "status": "ok",
                        "content": {"type": "text", "text": content},
                    },
                }
            )
        elif role == "assistant" and isinstance(message.get("content"), str):
            final_answer = message["content"].strip()
    if not instruction or len(steps) < 2 or pending or not final_answer:
        return None
    return {
        "instruction": instruction,
        "tools": [tool_definition(item) for item in tools],
        "steps": steps,
        "final_answer": final_answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/eval/tool_chain/bfcl_v1_3.jsonl")
    )
    parser.add_argument("--max-examples", type=int, default=128)
    args = parser.parse_args()

    candidates = sorted(args.source_dir.rglob("*.jsonl"))
    written = []
    rejected = 0
    for path in candidates:
        if "multi" not in path.name.lower() and "multi" not in str(path.parent).lower():
            continue
        for record in lines(path):
            try:
                row = normalize(record)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                row = None
            if row is None:
                rejected += 1
                continue
            written.append(row)
            if len(written) >= args.max_examples:
                break
        if len(written) >= args.max_examples:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in written:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    metadata = {
        "source": "ShishirPatil/gorilla Berkeley Function Calling Leaderboard",
        "source_release": "v1.3",
        "scope": "explicit multi-turn response paths only; no environment execution",
        "written": len(written),
        "rejected": rejected,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(written)} response-path examples to {args.output}")


if __name__ == "__main__":
    main()
