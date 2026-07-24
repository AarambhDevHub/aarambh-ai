#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - <<'PY'
import json
from pathlib import Path

training = [json.loads(line) for line in Path("data/agent_chain_smoke.jsonl").read_text().splitlines() if line.strip()]
replay = [json.loads(line) for line in Path("data/agent_results_smoke.jsonl").read_text().splitlines() if line.strip()]
evaluation = [json.loads(line) for line in Path("data/eval/tool_chain/data.jsonl").read_text().splitlines() if line.strip()]
assert len(training) == 1 and len(training[0]["turns"]) == 3
assert len(replay) == 3
assert [row["result"]["call_id"] for row in replay] == [f"call_{index:04d}" for index in range(1, 4)]
assert len(evaluation) == 1 and len(evaluation[0]["steps"]) == 3
print("Phase 37 fixtures: valid")
PY

cargo test --locked -p aarambh-ai-tokenizer tool_protocol
cargo test --locked -p aarambh-ai-agent
cargo test --locked -p aarambh-ai-finetune tool_sft
cargo test --locked -p aarambh-ai-eval tool_chain
cargo run --quiet --locked -p aarambh-ai -- agent --help | rg --quiet -- '--max-steps'

echo "Phase 37 source-only chain smoke completed"
