#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-data}
PHASE29_CHECKPOINT=${2:-checkpoints/wikitext103_medium_hybrid/model.safetensors}

scripts/phase16_prepare_longdoc.sh "$DATA_DIR"

if [[ ! -f "$PHASE29_CHECKPOINT" ]]; then
  echo "warning: Phase 29 checkpoint not found at $PHASE29_CHECKPOINT" >&2
  echo "set retrofit_from to an existing hybrid SafeTensors checkpoint" >&2
fi

echo "long-context corpus: $DATA_DIR/long_context/wikitext103_longdoc.txt"
echo "Phase 29 checkpoint: $PHASE29_CHECKPOINT"
echo "medium DSA config: configs/medium_hybrid_dsa.toml"
echo "large DSA config: configs/large_hybrid_dsa.toml"
