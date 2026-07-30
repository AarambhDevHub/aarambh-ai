#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <mtp-config> <mtp-model> <tokenizer>" >&2
  exit 2
fi

CONFIG=$1
MODEL=$2
TOKENIZER=$3
BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
PROMPT=${PROMPT:-"Explain why the sky appears blue."}
MAX_TOKENS=${MAX_TOKENS:-128}
RUNS=${RUNS:-5}

for path in "$CONFIG" "$MODEL" "$TOKENIZER"; do
  [[ -f "$path" ]] || {
    echo "missing input: $path" >&2
    exit 2
  }
done
[[ -x "$BIN" ]] || {
  echo "missing $BIN; build the release binary before benchmarking" >&2
  exit 2
}
rg -q '^\[model\.mtp\]' "$CONFIG" || {
  echo "config must contain [model.mtp]" >&2
  exit 2
}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

run_case() {
  local label=$1
  shift
  local run
  for run in $(seq 1 "$RUNS"); do
    "$BIN" infer \
      --config "$CONFIG" \
      --model "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --prompt "$PROMPT" \
      --max-tokens "$MAX_TOKENS" \
      --greedy --safety none --stats "$@" \
      >"$TMP_DIR/$label.$run.out" 2>"$TMP_DIR/$label.$run.stats"
    grep 'generation_stats' "$TMP_DIR/$label.$run.stats" |
      sed "s/^/case=$label run=$run /"
  done
}

run_case baseline
run_case mtp --speculative

for run in $(seq 1 "$RUNS"); do
  cmp "$TMP_DIR/baseline.$run.out" "$TMP_DIR/mtp.$run.out" || {
    echo "greedy output mismatch on run $run" >&2
    exit 1
  }
done

echo "all $RUNS greedy outputs matched exactly"
