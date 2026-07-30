#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <phase29-config> <phase29-model> <dsa-config> <dsa-model> <tokenizer>" >&2
  exit 2
fi

PHASE29_CONFIG=$1
PHASE29_MODEL=$2
DSA_CONFIG=$3
DSA_MODEL=$4
TOKENIZER=$5
BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
MAX_TOKENS=${MAX_TOKENS:-32}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-"4096 16384 32768"}

if [[ ! -x "$BIN" ]]; then
  echo "missing $BIN; build the release binary before benchmarking" >&2
  exit 2
fi

make_prompt() {
  local words=$1
  awk -v words="$words" 'BEGIN { for (i = 0; i < words; ++i) printf "context%d ", i % 97 }'
}

run_case() {
  local label=$1
  local config=$2
  local model=$3
  local words=$4
  local prompt
  local stats
  prompt=$(make_prompt "$words")
  stats=$(mktemp)
  "$BIN" infer \
    --config "$config" \
    --model "$model" \
    --tokenizer "$TOKENIZER" \
    --prompt "$prompt" \
    --max-tokens "$MAX_TOKENS" \
    --greedy --safety none --stats >/dev/null 2>"$stats"
  printf 'model=%s requested_context_words=%s ' "$label" "$words"
  grep -E 'generation_stats|dsa_cache_stats' "$stats" | tr '\n' ' '
  printf '\n'
  rm -f "$stats"
}

for context in $CONTEXT_LENGTHS; do
  run_case phase29 "$PHASE29_CONFIG" "$PHASE29_MODEL" "$context"
  run_case dsa "$DSA_CONFIG" "$DSA_MODEL" "$context"
done
