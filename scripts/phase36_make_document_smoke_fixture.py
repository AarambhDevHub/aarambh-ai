#!/usr/bin/env python3
"""Create four tiny two-page PDFs and Phase 36 document-QA records."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE20 = ROOT / "scripts" / "phase20_make_vqa_smoke_fixture.py"
OUT = ROOT / "data" / "document_smoke"
DOCUMENTS = OUT / "documents"
EVAL_OUT = ROOT / "data" / "eval" / "document_qa_smoke"


def load_phase20_module():
    spec = importlib.util.spec_from_file_location("phase20_make_vqa_smoke_fixture", PHASE20)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {PHASE20}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_pdf(path: Path, colors: list[tuple[float, float, float]]) -> None:
    objects: list[bytes] = []
    page_ids = [3 + index * 2 for index in range(len(colors))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()
    )
    for page_id, color in zip(page_ids, colors, strict=True):
        content_id = page_id + 1
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 128 96] "
                f"/Resources << >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        stream = (
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg "
            "0 0 128 96 re f\n"
        ).encode()
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"endstream"
        )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_id} 0 obj\n".encode())
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(pdf)


def main() -> None:
    phase20 = load_phase20_module()
    phase20.main()
    DOCUMENTS.mkdir(parents=True, exist_ok=True)
    examples = [
        ("red_invoice", "red", (1.0, 0.0, 0.0), (0.9, 0.9, 0.9)),
        ("green_form", "green", (0.0, 0.8, 0.0), (0.9, 0.9, 0.9)),
        ("blue_report", "blue", (0.0, 0.0, 1.0), (0.9, 0.9, 0.9)),
        ("yellow_table", "yellow", (1.0, 1.0, 0.0), (0.9, 0.9, 0.9)),
    ]
    for name, _answer, first, second in examples:
        write_pdf(DOCUMENTS / f"{name}.pdf", [first, second])

    records = [
        {
            "document": f"{name}.pdf",
            "question": "What color fills the first page?",
            "answers": [answer],
            "tags": ["table"] if name == "yellow_table" else [],
        }
        for name, answer, _first, _second in examples
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "document_qa_smoke_4.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    eval_records = [
        {
            **record,
            "document": f"../../document_smoke/documents/{record['document']}",
        }
        for record in records[:2]
    ]
    (EVAL_OUT / "data.jsonl").write_text(
        "\n".join(json.dumps(record) for record in eval_records) + "\n",
        encoding="utf-8",
    )
    print(f"document smoke data: {OUT / 'document_qa_smoke_4.jsonl'}")
    print(f"document smoke eval: {EVAL_OUT / 'data.jsonl'}")
    print(f"document smoke PDFs: {DOCUMENTS}")


if __name__ == "__main__":
    main()
