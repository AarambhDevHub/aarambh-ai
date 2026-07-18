#!/usr/bin/env python3
"""Create deterministic distillation prompt JSONL from a plaintext corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source plaintext corpus")
    parser.add_argument("output", type=Path, help="destination prompt JSONL")
    parser.add_argument("--max-prompts", type=int, default=10000)
    parser.add_argument("--min-chars", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=512)
    return parser.parse_args()


def corpus_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if line.startswith("=") and line.endswith("="):
            continue
        current.append(line)
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def main() -> None:
    args = parse_args()
    if args.max_prompts <= 0:
        raise SystemExit("--max-prompts must be greater than zero")
    if args.min_chars <= 0 or args.max_chars < args.min_chars:
        raise SystemExit("character bounds must satisfy 0 < min <= max")

    prompts = [
        paragraph[: args.max_chars].rstrip()
        for paragraph in corpus_paragraphs(args.input.read_text(encoding="utf-8"))
        if len(paragraph) >= args.min_chars
    ][: args.max_prompts]
    if not prompts:
        raise SystemExit("no source paragraphs satisfied the requested bounds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for index, prompt in enumerate(prompts):
            json.dump(
                {"id": f"prompt-{index:06d}", "prompt": prompt},
                output,
                ensure_ascii=False,
            )
            output.write("\n")
    print(f"wrote {len(prompts)} distillation prompts to {args.output}")


if __name__ == "__main__":
    main()
