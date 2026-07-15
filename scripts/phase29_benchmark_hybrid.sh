#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <dense-config> <dense-model> <hybrid-config> <hybrid-model> <tokenizer> [prompt]" >&2
  exit 2
fi

DENSE_CONFIG=$1
DENSE_MODEL=$2
HYBRID_CONFIG=$3
HYBRID_MODEL=$4
TOKENIZER=$5
PROMPT=${6:-"Summarize the most important facts in this document."}
BIN=${AARAMBH_BIN:-target/release/aarambh-ai}
RUNS=${RUNS:-3}
MAX_TOKENS=${MAX_TOKENS:-256}

if [[ ! -x "$BIN" ]]; then
  echo "missing release binary at $BIN; build with cargo build --release -p aarambh-ai --features cuda" >&2
  exit 2
fi

benchmark() {
  local label=$1
  local config=$2
  local model=$3
  local sum=0
  local tmp
  tmp=$(mktemp)
  for run in $(seq 1 "$RUNS"); do
    "$BIN" infer \
      --config "$config" \
      --model "$model" \
      --tokenizer "$TOKENIZER" \
      --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" \
      --greedy --safety none --stats >/dev/null 2>"$tmp"
    local rate
    rate=$(sed -n 's/.* tok_s=\([^ ]*\).*/\1/p' "$tmp" | tail -1)
    if [[ -z "$rate" ]]; then
      echo "unable to read tok_s for $label run $run" >&2
      rm -f "$tmp"
      exit 1
    fi
    sum=$(awk -v total="$sum" -v value="$rate" 'BEGIN { printf "%.6f", total + value }')
    printf 'model=%s run=%s tok_s=%s\n' "$label" "$run" "$rate"
  done
  rm -f "$tmp"
  awk -v total="$sum" -v runs="$RUNS" 'BEGIN { printf "%.3f", total / runs }'
}

dense_avg=$(benchmark dense "$DENSE_CONFIG" "$DENSE_MODEL" | tee /dev/stderr | tail -1 | sed 's/.*tok_s=//')
hybrid_avg=$(benchmark hybrid "$HYBRID_CONFIG" "$HYBRID_MODEL" | tee /dev/stderr | tail -1 | sed 's/.*tok_s=//')
speedup=$(awk -v dense="$dense_avg" -v hybrid="$hybrid_avg" \
  'BEGIN { if (dense == 0) print "0.000"; else printf "%.3f", hybrid / dense }')

echo "dense_average_tok_s=$dense_avg"
echo "hybrid_average_tok_s=$hybrid_avg"
echo "hybrid_speedup=${speedup}x"
