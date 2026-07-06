#!/usr/bin/env python3
"""Create a tiny local Phase 19 vision smoke-test fixture.

The fixture is intentionally synthetic. It verifies that the image path,
CLIP-style encoder, projector training, checkpointing, and image inference
pipeline are wired correctly without downloading full COCO or CLIP assets.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "vision_smoke"
CHECKPOINT_OUT = ROOT / "checkpoints" / "vision_projector_smoke"
SOURCE_TOKENIZER = ROOT / "checkpoints" / "tiny_shakespeare" / "tokenizer.json"


SPECIALS = {
    "<|endoftext|>": 0,
    "<|pad|>": 1,
    "<|bos|>": 2,
    "<think>": 3,
    "</think>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<image>": 7,
    "<image_end>": 8,
}


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    data = bytearray()
    header: dict[str, object] = {}
    for name in sorted(tensors):
        tensor = np.ascontiguousarray(tensors[name], dtype=np.float32)
        start = len(data)
        payload = tensor.tobytes(order="C")
        data.extend(payload)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [start, start + len(payload)],
        }

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(header_bytes) % 8)) % 8
    header_bytes += b" " * padding
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + data)


def make_clip_fixture() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    config = {
        "patch_size": 4,
        "image_size": 8,
        "in_channels": 3,
        "vit_d_model": 16,
        "vit_layers": 1,
        "vit_heads": 4,
        "mlp_dim": 32,
        "num_patches": 4,
        "norm_eps": 0.00001,
    }
    (OUT / "clip_tiny_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    width = config["vit_d_model"]
    mlp = config["mlp_dim"]
    patch_dim = config["in_channels"] * config["patch_size"] * config["patch_size"]

    def randn(shape: tuple[int, ...], scale: float = 0.02) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def zeros(shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

    def ones(shape: tuple[int, ...]) -> np.ndarray:
        return np.ones(shape, dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "patch_embed.weight": randn((width, patch_dim)),
        "class_embedding": randn((width,)),
        "position_embedding": randn((config["num_patches"] + 1, width), 0.01),
        "pre_norm.weight": ones((width,)),
        "pre_norm.bias": zeros((width,)),
        "post_norm.weight": ones((width,)),
        "post_norm.bias": zeros((width,)),
        "blocks.0.norm1.weight": ones((width,)),
        "blocks.0.norm1.bias": zeros((width,)),
        "blocks.0.norm2.weight": ones((width,)),
        "blocks.0.norm2.bias": zeros((width,)),
        "blocks.0.attn.q_proj.weight": randn((width, width)),
        "blocks.0.attn.q_proj.bias": zeros((width,)),
        "blocks.0.attn.k_proj.weight": randn((width, width)),
        "blocks.0.attn.k_proj.bias": zeros((width,)),
        "blocks.0.attn.v_proj.weight": randn((width, width)),
        "blocks.0.attn.v_proj.bias": zeros((width,)),
        "blocks.0.attn.out_proj.weight": randn((width, width)),
        "blocks.0.attn.out_proj.bias": zeros((width,)),
        "blocks.0.mlp.fc1.weight": randn((mlp, width)),
        "blocks.0.mlp.fc1.bias": zeros((mlp,)),
        "blocks.0.mlp.fc2.weight": randn((width, mlp)),
        "blocks.0.mlp.fc2.bias": zeros((width,)),
    }
    write_safetensors(OUT / "clip_tiny.safetensors", tensors)


def make_images_and_captions() -> None:
    image_dir = OUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    examples = [
        ("red_square.png", (220, 40, 40), "A red square."),
        ("green_square.png", (40, 180, 70), "A green square."),
        ("blue_square.png", (50, 80, 220), "A blue square."),
        ("yellow_square.png", (220, 190, 40), "A yellow square."),
    ]
    lines = []
    for filename, color, caption in examples:
        image = Image.new("RGB", (32, 32), color)
        image.save(image_dir / filename)
        lines.append(json.dumps({"image": filename, "caption": caption}))
    (OUT / "train_smoke_4.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_smoke_tokenizer() -> None:
    if not SOURCE_TOKENIZER.exists():
        raise SystemExit(f"missing source tokenizer: {SOURCE_TOKENIZER}")

    CHECKPOINT_OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = json.loads(SOURCE_TOKENIZER.read_text(encoding="utf-8"))
    vocab = tokenizer["model"]["vocab"]
    remapped: dict[str, int] = dict(SPECIALS)

    for token, token_id in vocab.items():
        if token in SPECIALS:
            continue
        if token_id < 7:
            continue
        shifted = token_id + 2
        if shifted < len(vocab):
            remapped[token] = shifted

    tokenizer["model"]["vocab"] = remapped
    (CHECKPOINT_OUT / "tokenizer.json").write_text(
        json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    make_clip_fixture()
    make_images_and_captions()
    make_smoke_tokenizer()
    print(f"vision smoke fixture: {OUT}")
    print(f"vision smoke tokenizer: {CHECKPOINT_OUT / 'tokenizer.json'}")


if __name__ == "__main__":
    main()
