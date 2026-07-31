#!/usr/bin/env bash
# Phase 41 — Multi-Head Latent Attention (MLA) retrofit data preparation.
#
# Same continued-pretraining corpus style as Phase 29: long documents, since
# MLA's KV-cache payoff (like Gated DeltaNet's) shows up at long context, not
# short prompts. This script reuses the Phase 16 long-document packer and only
# points the user at the MLA hybrid configs and the base checkpoint to retrofit
# from. No new dataset is downloaded; the retrofit corpus is the same
# long-context wikitext pack Phase 29 already produced.
set -euo pipefail

DATA_DIR=${1:-data}
BASE_CHECKPOINT=${2:-checkpoints/wikitext103_medium_hybrid/model.safetensors}

# Reuse the Phase 16 long-document packer (idempotent if already prepared).
scripts/phase16_prepare_longdoc.sh "$DATA_DIR"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "warning: base v3 checkpoint not found at $BASE_CHECKPOINT" >&2
  echo "set retrofit_from in the MLA hybrid config to an existing v3 SafeTensors checkpoint" >&2
  echo "the retrofit path reinitialises the scheduled MLA layers and loads every other layer unchanged" >&2
fi

echo "MLA retrofit corpus: $DATA_DIR/long_context/wikitext103_longdoc.txt"
echo "base v3 checkpoint:  $BASE_CHECKPOINT"
echo "medium MLA config:   configs/medium_hybrid_mla.toml"
echo "large MLA config:    configs/large_hybrid_mla.toml"
echo "smoke config:        configs/mla_smoke.toml"
