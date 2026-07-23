#!/usr/bin/env python3
"""Normalize user-downloaded DocVQA or MP-DocVQA annotations to Phase 36 JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, list):
        yield from value
        return
    if not isinstance(value, dict):
        raise SystemExit("annotation root must be an array or object")
    for key in ("data", "questions", "examples"):
        if isinstance(value.get(key), list):
            yield from value[key]
            return
    raise SystemExit("could not find a data/questions/examples array")


def first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, "", []):
            return value
    return None


def normalize(record: dict[str, Any], documents_dir: Path) -> dict[str, Any]:
    question = first(record, "question", "query")
    answers = first(record, "answers", "answer")
    if isinstance(answers, str):
        answers = [answers]
    if not isinstance(question, str) or not isinstance(answers, list) or not answers:
        raise ValueError("record requires question and answer(s)")

    page_paths = first(record, "page_paths", "page_images", "pages_paths")
    document = first(
        record,
        "document_path",
        "document",
        "image",
        "image_path",
        "pdf",
    )
    if page_paths:
        if not isinstance(page_paths, list):
            raise ValueError("page_paths must be an array")
        source = {"page_paths": [str(documents_dir / Path(path)) for path in page_paths]}
    elif document:
        source = {"document": str(documents_dir / Path(str(document)))}
    else:
        raise ValueError("record requires a document/image path or page_paths")

    output = {
        **source,
        "question": question.strip(),
        "answers": [str(answer).strip() for answer in answers if str(answer).strip()],
    }
    pages = first(record, "page_numbers", "pages")
    if isinstance(pages, list) and all(isinstance(page, int) for page in pages):
        output["pages"] = pages
    tags = record.get("tags")
    if isinstance(tags, list):
        output["tags"] = [str(tag) for tag in tags]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--documents-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    payload = json.loads(args.annotations.read_text(encoding="utf-8"))
    output = []
    skipped = 0
    for record in records(payload):
        try:
            output.append(normalize(record, args.documents_dir))
        except ValueError:
            skipped += 1
        if args.max_samples is not None and len(output) >= args.max_samples:
            break
    if not output:
        raise SystemExit("no valid document-QA records were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in output) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(output)} records to {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()
