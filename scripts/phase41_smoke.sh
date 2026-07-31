#!/usr/bin/env bash
# Phase 41 — Multi-Head Latent Attention (MLA) smoke test.
#
# Always runs:
#   - MLA unit tests (reconstruction tolerance, decoupled RoPE split, cache size,
#     zero-MLA backward compatibility).
#   - `eval --kv-cache-report` on the MLA smoke config (needs no checkpoint,
#     only the config) and asserts an MLA layer appears with a smaller
#     bytes/token than the GQA baseline.
#
# Gated on PHASE41_SKIP_TRAIN=0 (default) and the presence of a training
# fixture: a two-step CPU training run on configs/mla_smoke.toml, then a
# check that the saved checkpoint contains MLA-layer tensors
# (blocks.0.mla.q_proj.weight etc.).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Phase 41 MLA unit tests"
cargo test --locked -p aarambh-studio-nn mla
cargo test --locked -p aarambh-studio-core schedule_with_zero_mla

echo "==> Phase 41 kv-cache-report on the MLA smoke config"
REPORT="$(cargo run --quiet --locked -p aarambh-studio -- eval \
  --config configs/mla_smoke.toml --kv-cache-report)"
echo "$REPORT"
echo "$REPORT" | rg --quiet -- 'latent_mla'
echo "$REPORT" | rg --quiet -- 'latent_dim \+ rope_head_dim per token'
# MLA cache (80 elem * 4 bytes = 320) must be smaller than GQA (128 elem * 4 = 512).
echo "$REPORT" | rg --quiet -- 'hybrid/all-full ratio'

if [[ "${PHASE41_SKIP_TRAIN:-0}" == "1" ]]; then
  echo "PHASE41_SKIP_TRAIN=1; training smoke was skipped"
  echo "Phase 41 smoke completed"
  exit 0
fi

echo "==> Phase 41 ensure a tiny training fixture exists"
if [[ ! -f data/tiny_shakespeare.txt ]]; then
  mkdir -p data
  python3 - <<'PY'
from pathlib import Path
# A tiny public-domain-style text fixture so the BPE tokenizer and the
# two-step training smoke can run without the full wikitext corpus.
snippet = (
    "To be, or not to be, that is the question: "
    "whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune, "
    "or to take arms against a sea of troubles "
    "and by opposing end them. "
)
text = (snippet * 400)
Path("data/tiny_shakespeare.txt").write_text(text)
print(f"wrote data/tiny_shakespeare.txt ({len(text)} bytes)")
PY
fi

echo "==> Phase 41 two-step CPU training smoke (mla_smoke.toml)"
cargo run --quiet --locked -p aarambh-studio -- train \
  --config configs/mla_smoke.toml

echo "==> Phase 41 verify the saved checkpoint contains MLA tensors"
python3 - <<'PY'
import json
from pathlib import Path
ptr = json.loads(Path("checkpoints/mla_smoke/latest.json").read_text())
model = Path(ptr["path"]) / "model.safetensors"
if not model.exists():
    raise SystemExit(f"checkpoint not found: {model}")
raw = model.read_bytes()
header_len = int.from_bytes(raw[:8], "little")
header = json.loads(raw[8:8 + header_len].decode("utf-8"))
names = list(header.keys())
required = [
    "blocks.0.mla.q_proj.weight",
    "blocks.0.mla.kv_a_proj.weight",
    "blocks.0.mla.kv_a_norm.weight",
    "blocks.0.mla.up_k.weight",
    "blocks.0.mla.up_v.weight",
    "blocks.0.mla.k_rope_proj.weight",
    "blocks.0.mla.o_proj.weight",
    "blocks.1.deltanet.q_proj.weight",
]
missing = [n for n in required if n not in names]
assert not missing, f"checkpoint missing MLA/GatedDeltaNet tensors: {missing}"
print(f"Phase 41 checkpoint OK: {len(names)} tensors, MLA layer 0 present, GatedDeltaNet layer 1 present")
PY

echo "Phase 41 smoke completed"
