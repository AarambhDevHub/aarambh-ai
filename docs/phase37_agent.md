# Phase 37: Long-Horizon Tool-Use Chains

Phase 37 turns the Phase 26 single-call tool decoder into a bounded loop. The
model emits a schema-valid call, the caller executes it, Aarambh ingests the
typed result, and the model decides whether to call again or answer.

## Security Boundary

Aarambh does not execute tools. It does not run a shell, make a network request,
load a plugin, or authorize an action. The embedding application performs those
operations and returns one JSON result. Treat every generated call as
untrusted input and enforce authorization outside Aarambh.

Strict safety is the CLI default. The initial prompt, result text/error or media
description, each structured model decision, and the final answer are inspected
with the existing safety policy and audit log.

## Result Protocol

Interactive mode reads one JSON object per line from stdin. Successful text:

```json
{"call_id":"call_0001","status":"ok","content":{"type":"text","text":"18 C"}}
```

External failure:

```json
{"call_id":"call_0002","status":"error","error":"service unavailable"}
```

Native media results use `type` equal to `image`, `video`, or `document`:

```json
{"call_id":"call_0003","status":"ok","content":{"type":"document","path":"retrieved/report.pdf","pages":[1,3],"description":"Quarterly report"}}
```

Paths must exist under `--result-root`. Text/error payloads are limited to
64 KiB, descriptions to 4 KiB, and document pages must be unique and one-based.
The call id must exactly match the pending request.

## Run

Build a checkpoint trained on multi-step tool traces, then run:

```sh
target/release/aarambh-ai agent \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tool_sft/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tools data/agent_tools_smoke.json \
  --prompt "Find the shipping price for customer C-17's latest order." \
  --max-steps 8 --greedy \
  --safety strict \
  --result-root .
```

The CLI writes each call to stderr. Execute it externally, then enter one result
line. `--jsonl` emits lifecycle events on stdout. For deterministic replay:

```sh
target/release/aarambh-ai agent \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tool_sft/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tools data/agent_tools_smoke.json \
  --results data/agent_results_smoke.jsonl \
  --prompt "Find the shipping price for customer C-17's latest order." \
  --max-steps 8 --jsonl
```

Replay entries may include `expected_call`; a name or argument mismatch fails
immediately. This makes response paths deterministic for tests and evaluation.

## Context And Media

The default policy is `--eviction drop-oldest --keep-recent 4`. It removes the
oldest complete call/result exchange before context overflow while preserving
the recent protected turns. `--eviction summarise --summary-tokens 128` asks
the model to compact evicted factual state and rebuilds the tool-aware prefix.
If protected state still cannot fit, generation fails instead of truncating a
live exchange.

An image, video, or document result is projected natively for the immediately
following decision. Later turns retain only its bounded path, pages, and
description metadata. Only one media result is active at a time.

## Multi-Step SFT

Existing `finetune tool-sft` and `tool-qlora` commands auto-detect records with
a `turns` array. See `data/agent_chain_smoke.jsonl`. Every turn contains:

```json
{"tool_call":{"name":"lookup","arguments":{"key":"x"}},"tool_result":{"status":"ok","content":{"type":"text","text":"value"}}}
```

Loss covers each tool-call span and the final response. Initial prompts,
caller results, and optional thinking fields are context-only. Long traces that
would be truncated are rejected.

## Evaluation And BFCL

Run the checked-in three-call response path:

```sh
target/release/aarambh-ai eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tool_sft/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks tool-chain \
  --data-dir data/eval \
  --agent-max-steps 8 \
  --max-new-tokens 128
```

Prepare compatible explicit response paths from a local BFCL v1.3 checkout:

```sh
python3 scripts/phase37_prepare_bfcl_multiturn.py \
  --source-dir /path/to/berkeley-function-call-leaderboard \
  --output data/eval/tool_chain/bfcl_v1_3.jsonl
```

This evaluates the model's response path against scripted results. It does not
execute BFCL environments and does not represent full BFCL stateful coverage.
Run `scripts/phase37_smoke.sh` for source/protocol tests. Useful benchmark
accuracy requires real multi-step training and is not implied by the fixture.
