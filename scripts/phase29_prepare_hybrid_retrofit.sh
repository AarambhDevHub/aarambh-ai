#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-data}
BASE_CHECKPOINT=${2:-checkpoints/wikitext103_medium/model.safetensors}

scripts/phase16_prepare_longdoc.sh "$DATA_DIR"

if [[ ! -f "$BASE_CHECKPOINT" ]]; then
  echo "warning: base checkpoint not found at $BASE_CHECKPOINT" >&2
  echo "set retrofit_from in the hybrid config to an existing dense v2 SafeTensors checkpoint" >&2
fi

echo "long-context retrofit corpus: $DATA_DIR/long_context/wikitext103_longdoc.txt"
echo "base checkpoint: $BASE_CHECKPOINT"
echo "medium config: configs/wikitext103_medium_hybrid.toml"
echo "large config: configs/wikitext103_large_hybrid.toml"
