#!/usr/bin/env python3
"""Create a tiny local Phase 20 VQA smoke-test fixture."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE19 = ROOT / "scripts" / "phase19_make_smoke_fixture.py"
OUT = ROOT / "data" / "vision_smoke"
EVAL_OUT = ROOT / "data" / "eval" / "vqa_smoke"


def load_phase19_module():
    spec = importlib.util.spec_from_file_location("phase19_make_smoke_fixture", PHASE19)
    if spec is None or spec.loader is None:
        raise SystemExit(f"failed to load {PHASE19}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_vqa_data() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    examples = [
        {"image": "red_square.png", "question": "What color is the square?", "answer": "red"},
        {
            "image": "green_square.png",
            "question": "What color is the square?",
            "answer": "green",
        },
        {"image": "blue_square.png", "question": "What color is the square?", "answer": "blue"},
        {
            "image": "yellow_square.png",
            "question": "What color is the square?",
            "answer": "yellow",
            "thinking": "The square is yellow.",
        },
    ]
    (OUT / "vqa_smoke_4.jsonl").write_text(
        "\n".join(json.dumps(item) for item in examples) + "\n", encoding="utf-8"
    )

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    eval_examples = [
        {
            "image": "../../vision_smoke/images/red_square.png",
            "question": "What color is the square?",
            "answers": ["red"],
            "keywords": ["red"],
        },
        {
            "image": "../../vision_smoke/images/blue_square.png",
            "question": "What color is the square?",
            "answers": ["blue"],
            "keywords": ["blue"],
        },
    ]
    (EVAL_OUT / "data.jsonl").write_text(
        "\n".join(json.dumps(item) for item in eval_examples) + "\n", encoding="utf-8"
    )


def write_projector_init(phase19_module) -> None:
    rng = np.random.default_rng(123)

    def randn(shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    tensors = {
        "fc1.weight": randn((384, 16)),
        "fc1.bias": zeros((384,)),
        "fc2.weight": randn((384, 384)),
        "fc2.bias": zeros((384,)),
    }
    phase19_module.write_safetensors(OUT / "projector_init.safetensors", tensors)


def main() -> None:
    phase19 = load_phase19_module()
    phase19.main()
    write_vqa_data()
    write_projector_init(phase19)
    print(f"VQA smoke data: {OUT / 'vqa_smoke_4.jsonl'}")
    print(f"VQA smoke eval: {EVAL_OUT / 'data.jsonl'}")
    print(f"Projector init: {OUT / 'projector_init.safetensors'}")


if __name__ == "__main__":
    main()
