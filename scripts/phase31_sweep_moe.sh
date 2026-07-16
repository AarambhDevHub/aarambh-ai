#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <scratch-base-config> <eval-data-dir> <output-dir>" >&2
  exit 2
fi

BASE_CONFIG=$1
DATA_DIR=$2
OUTPUT_DIR=$3
BIN=${AARAMBH_BIN:-target/release/aarambh-ai}
TASKS=${TASKS:-ppl}
MAX_EXAMPLES=${MAX_EXAMPLES:-128}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
MAX_STEPS=${MAX_STEPS:-}
CASES=${CASES:-"coarse_8x1:8:1:2:0 fine_8x2:8:2:4:1 fine_8x4:8:4:8:1 wide_16x4:16:4:8:1"}

[[ -f "$BASE_CONFIG" ]] || {
  echo "missing base config: $BASE_CONFIG" >&2
  exit 2
}
[[ -d "$DATA_DIR" ]] || {
  echo "missing eval data directory: $DATA_DIR" >&2
  exit 2
}
[[ -x "$BIN" ]] || {
  echo "missing $BIN; build the release binary before running the sweep" >&2
  exit 2
}
command -v python3 >/dev/null 2>&1 || {
  echo "phase31 sweep requires python3" >&2
  exit 2
}
if rg -q '^retrofit_from\s*=|^\[moe_retrofit\]' "$BASE_CONFIG"; then
  echo "the sweep base must train from scratch; retrofit settings are not accepted" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR/configs" "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/scores"
SUMMARY="$OUTPUT_DIR/results.md"
printf '%s\n\n' '# Phase 31 MoE Sweep' >"$SUMMARY"
printf '%s\n' '| Case | Coarse groups | Split factor | Routed experts | Top-k | Shared experts |' >>"$SUMMARY"
printf '%s\n' '|---|---:|---:|---:|---:|---:|' >>"$SUMMARY"

baseline_score=""
for spec in $CASES; do
  IFS=: read -r name num_experts factor top_k shared <<<"$spec"
  if [[ -z "$name" || -z "$shared" ]]; then
    echo "invalid CASES entry: $spec" >&2
    exit 2
  fi

  config="$OUTPUT_DIR/configs/$name.toml"
  checkpoint_dir="$OUTPUT_DIR/checkpoints/$name"
  python3 - "$BASE_CONFIG" "$config" "$checkpoint_dir" \
    "$num_experts" "$factor" "$top_k" "$shared" "$MAX_STEPS" <<'PY'
import json
import pathlib
import re
import sys

source, output, checkpoint, experts, factor, top_k, shared, max_steps = sys.argv[1:]
text = pathlib.Path(source).read_text(encoding="utf-8")

def set_section_key(document, section, key, value):
    pattern = re.compile(
        rf"(?ms)(^\[{re.escape(section)}\]\s*\n)(.*?)(?=^\[|\Z)"
    )
    match = pattern.search(document)
    if match is None:
        raise SystemExit(f"missing [{section}] in {source}")
    body = match.group(2)
    key_pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    replacement = f"{key} = {value}"
    if key_pattern.search(body):
        body = key_pattern.sub(replacement, body)
    else:
        body = body.rstrip() + "\n" + replacement + "\n"
    return document[: match.start(2)] + body + document[match.end(2) :]

expert_width_match = re.search(
    r"(?ms)^\[model\.moe\]\s*\n.*?^expert_ffn_dim\s*=\s*(\d+)", text
)
if expert_width_match is None:
    raise SystemExit(f"missing model.moe.expert_ffn_dim in {source}")
expert_width = int(expert_width_match.group(1))
if expert_width % int(factor) != 0:
    raise SystemExit(
        f"expert_ffn_dim={expert_width} is not divisible by split factor {factor}"
    )

for key, value in (
    ("num_experts", experts),
    ("fine_grained_factor", factor),
    ("top_k", top_k),
    ("num_shared_experts", shared),
):
    text = set_section_key(text, "model.moe", key, value)
text = set_section_key(text, "train", "checkpoint_dir", json.dumps(checkpoint))
if max_steps:
    text = set_section_key(text, "train", "max_steps", max_steps)
text = re.sub(r"(?m)^resume\s*=.*$", "resume = false", text)
pathlib.Path(output).write_text(text, encoding="utf-8")
PY

  routed=$((num_experts * factor))
  printf '| %s | %s | %s | %s | %s | %s |\n' \
    "$name" "$num_experts" "$factor" "$routed" "$top_k" "$shared" >>"$SUMMARY"

  echo "training Phase 31 sweep case: $name"
  "$BIN" train --config "$config" 2>&1 | tee "$OUTPUT_DIR/$name.train.log"

  score="$OUTPUT_DIR/scores/$name.json"
  markdown="$OUTPUT_DIR/scores/$name.md"
  "$BIN" eval \
    --config "$config" \
    --tasks "$TASKS" \
    --data-dir "$DATA_DIR" \
    --max-examples "$MAX_EXAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --out "$score" \
    --markdown "$markdown"

  if [[ -z "$baseline_score" ]]; then
    baseline_score="$score"
    printf '\n%s\n\n' "## Baseline: $name" >>"$SUMMARY"
    cat "$markdown" >>"$SUMMARY"
  else
    comparison="$OUTPUT_DIR/scores/${name}_vs_baseline.md"
    "$BIN" eval --compare "$baseline_score" "$score" --markdown "$comparison"
    printf '\n%s\n\n' "## $name vs baseline" >>"$SUMMARY"
    cat "$comparison" >>"$SUMMARY"
  fi
done

echo "Phase 31 sweep results written to $SUMMARY"
