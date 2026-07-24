#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_EXAMPLES="${1:-128}"

scripts/phase17_prepare_eval_sets.sh data/eval "$MAX_EXAMPLES"
python3 scripts/phase20_make_vqa_smoke_fixture.py
python3 scripts/phase35_make_video_smoke_fixture.py
python3 scripts/phase36_make_document_smoke_fixture.py

test -f data/eval/forgetting/probes.json
test -f data/eval/tool_calling/data.jsonl
test -f data/eval/tool_chain/data.jsonl

echo "Phase 38 capability-probe data is ready under data/eval"
