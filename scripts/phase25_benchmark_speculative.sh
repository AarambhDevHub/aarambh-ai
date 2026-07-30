#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 <target-config> <target-model> <tokenizer> <draft-config> <draft-model> [prompt]" >&2
  exit 2
fi

TARGET_CONFIG=$1
TARGET_MODEL=$2
TOKENIZER=$3
DRAFT_CONFIG=$4
DRAFT_MODEL=$5
PROMPT=${6:-"To be, or not to be"}
BIN=${AARAMBH_STUDIO_BIN:-target/release/aarambh-studio}
RUNS=${RUNS:-3}
MAX_TOKENS=${MAX_TOKENS:-128}
DRAFT_TOKENS=${DRAFT_TOKENS:-4}

if [[ ! -x "$BIN" ]]; then
  echo "missing release binary at $BIN; run cargo build --release -p aarambh-studio --features cuda" >&2
  exit 2
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

run_target() {
  "$BIN" infer \
    --config "$TARGET_CONFIG" \
    --model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --greedy --safety none --stats
}

run_speculative() {
  "$BIN" infer \
    --config "$TARGET_CONFIG" \
    --model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --greedy --safety none --stats \
    --speculative \
    --draft-config "$DRAFT_CONFIG" \
    --draft-model "$DRAFT_MODEL" \
    --draft-tokens "$DRAFT_TOKENS"
}

echo "warming target and draft models..." >&2
run_target >"$tmp_dir/warm_target.out" 2>"$tmp_dir/warm_target.err"
run_speculative >"$tmp_dir/warm_spec.out" 2>"$tmp_dir/warm_spec.err"

target_sum=0
speculative_sum=0
for run in $(seq 1 "$RUNS"); do
  run_target >"$tmp_dir/target_$run.out" 2>"$tmp_dir/target_$run.err"
  run_speculative >"$tmp_dir/spec_$run.out" 2>"$tmp_dir/spec_$run.err"
  if ! cmp -s "$tmp_dir/target_$run.out" "$tmp_dir/spec_$run.out"; then
    echo "error: greedy target and speculative outputs differ on run $run" >&2
    diff -u "$tmp_dir/target_$run.out" "$tmp_dir/spec_$run.out" >&2 || true
    exit 1
  fi
  target_rate=$(sed -n 's/.* tok_s=\([^ ]*\).*/\1/p' "$tmp_dir/target_$run.err" | tail -1)
  speculative_rate=$(sed -n 's/.* tok_s=\([^ ]*\).*/\1/p' "$tmp_dir/spec_$run.err" | tail -1)
  acceptance=$(sed -n 's/.* acceptance_rate=\([^ ]*\).*/\1/p' "$tmp_dir/spec_$run.err" | tail -1)
  target_sum=$(awk -v sum="$target_sum" -v rate="$target_rate" 'BEGIN { printf "%.6f", sum + rate }')
  speculative_sum=$(awk -v sum="$speculative_sum" -v rate="$speculative_rate" 'BEGIN { printf "%.6f", sum + rate }')
  printf 'run=%s target_tok_s=%s speculative_tok_s=%s acceptance=%s\n' \
    "$run" "$target_rate" "$speculative_rate" "$acceptance"
done

target_avg=$(awk -v sum="$target_sum" -v runs="$RUNS" 'BEGIN { printf "%.3f", sum / runs }')
speculative_avg=$(awk -v sum="$speculative_sum" -v runs="$RUNS" 'BEGIN { printf "%.3f", sum / runs }')
speedup=$(awk -v target="$target_avg" -v speculative="$speculative_avg" \
  'BEGIN { if (target == 0) print "0.000"; else printf "%.3f", speculative / target }')

echo "average_target_tok_s=$target_avg"
echo "average_speculative_tok_s=$speculative_avg"
echo "speedup=${speedup}x"
echo "roadmap_target=1.8x (reported only; not a CI gate)"
