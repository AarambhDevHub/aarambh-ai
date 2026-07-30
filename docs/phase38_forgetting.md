# Phase 38: Forgetting Diagnostics

Phase 38 measures whether named model capabilities regress across checkpoints,
training steps, or committed self-learning updates. It is diagnostic only:
probes do not call backward, update an optimizer, alter replay, or mutate model
parameters.

## Contract

The checked-in manifest at `data/eval/forgetting/probes.json` maps eight
capabilities onto existing evaluation tasks:

| Capability | Existing task subset |
|---|---|
| Math | GSM8K subset |
| Code | HumanEval-lite |
| Reasoning | HellaSwag |
| Factual | MMLU-lite |
| Vision | VQA |
| Video | video QA |
| Document | document QA |
| Tool use | tool calling and tool chain |

No benchmark examples are duplicated in the manifest. A run-level
`--max-examples` may reduce, but never increase, each manifest cap. Baseline
and current task names, metrics, and example counts must match. A run records a typed
skip when data, modality configuration, or code-execution permission is
unavailable. Use `--require-all-probes` only when all eight probes must run.
HumanEval-lite additionally requires `--allow-code-exec`.

The default absolute significance threshold is `0.02`. For every capability:

```text
delta       = score_after - score_before
significant = abs(delta) >= threshold
forgotten   = significant and delta < 0
improved    = significant and delta > 0
```

The persistent store includes the probe-manifest fingerprint, tokenizer
fingerprint, suite ID, threshold, task-level scores, skips, and optional MoE
routing signatures. Reusing an ID with identical data is idempotent; reusing it
with different data is rejected.

## Prepare Probe Data

The text subsets use the Phase 17 downloader. The multimodal scripts generate
small local mechanism fixtures; replace them with held-out real datasets for
quality claims.

```sh
scripts/phase38_prepare_probes.sh 128
```

This command requires the Python `datasets` package and network access for the
public text datasets. Code probes remain disabled unless execution is
explicitly allowed.

## Compare Named Checkpoints

Record the baseline:

```sh
cargo run --release -p aarambh-studio -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks gsm8k \
  --data-dir data/eval \
  --max-examples 16 \
  --max-new-tokens 32 \
  --forgetting-manifest data/eval/forgetting/probes.json \
  --forgetting-store checkpoints/forgetting/curves.json \
  --checkpoint-id phase37-baseline \
  --out checkpoints/forgetting/baseline-scorecard.json
```

Record and compare a later checkpoint:

```sh
cargo run --release -p aarambh-studio -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000100/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks gsm8k \
  --data-dir data/eval \
  --max-examples 16 \
  --max-new-tokens 32 \
  --forgetting-manifest data/eval/forgetting/probes.json \
  --forgetting-store checkpoints/forgetting/curves.json \
  --checkpoint-id phase38-current \
  --baseline-id phase37-baseline \
  --forgetting-jsonl checkpoints/forgetting/manas-v1.jsonl \
  --out checkpoints/forgetting/current-scorecard.json \
  --markdown checkpoints/forgetting/current-scorecard.md
```

`--tasks` still supplies scorecard metadata, but the forgetting run is governed
by the validated manifest. A comparison requires the baseline to contain every
capability scored by the current run.

## Observe Training

Add this optional table to a normal training configuration:

```toml
[forgetting]
enabled = true
manifest = "data/eval/forgetting/probes.json"
data_dir = "data/eval"
store = "checkpoints/forgetting/curves.json"
jsonl = "checkpoints/forgetting/manas-v1.jsonl"
every_n_steps = 1000
max_examples = 16
max_new_tokens = 64
agent_max_steps = 8
significance_threshold = 0.02
allow_code_exec = false
require_all_probes = false
# baseline_id = "optional-existing-baseline"
```

Then run the existing command:

```sh
aarambh-studio train --config configs/your_training_config.toml
```

The observer records the run start, every configured optimizer-step cadence,
and the final step if it was not already recorded. It reads the live model
without writing an intermediate checkpoint. In distributed training all ranks
synchronize while rank 0 evaluates.

`configs/forgetting_smoke.toml` is a CPU mechanism configuration. Its tiny
random/short-trained model cannot establish useful capability quality.

## Observe Self-Learning

Attach diagnostics to a text self-learning turn:

```sh
aarambh-studio selflearn start \
  --mode text \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --prompt "What is 2 + 2?" \
  --self-learn-verifier math \
  --self-learn-ground-truth "#### 4" \
  --forgetting-manifest data/eval/forgetting/probes.json \
  --forgetting-store adapters/selflearn/forgetting_curves.json \
  --forgetting-jsonl adapters/selflearn/manas-v1.jsonl
```

The hook first captures a session baseline. GPU inline updates are probed after
commit. CPU deferred gradients are probed only when
`selflearn flush-gradients` commits them. Replay fine-tuning is probed after its
optimizer update. The evaluated model is an in-memory merged adapter view; the
base checkpoint is not overwritten.

Print stored curves:

```sh
aarambh-studio selflearn forgetting-report \
  --forgetting-store adapters/selflearn/forgetting_curves.json
```

## MoE Routing Drift

For MoE models, each textual probe example records the sorted top-k routed
expert set for every MoE layer. Comparisons report the fraction of matched
examples whose routed set changed. Routing drift is an early-warning diagnostic,
not proof of capability loss. Dense models do not collect routing signatures.

## Manas Bridge

`schemas/manas-forgetting-v1.schema.json` defines exactly seven fields:

```json
{
  "capability_or_concept": "math",
  "baseline_checkpoint_or_session": "phase37-baseline",
  "current_checkpoint_or_session": "phase38-current",
  "score_before": 0.42,
  "score_after": 0.39,
  "delta": -0.03,
  "significant": true
}
```

The export is newline-delimited JSON. Aarambh does not link to, discover, or
write into the sibling Manas repository. Moving or importing the JSONL is an
explicit operator action, so both projects remain independently buildable.

## Validation

Run source/contract checks:

```sh
scripts/phase38_smoke.sh
```

To include a real two-point checkpoint execution smoke:

```sh
PHASE38_MODEL=checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  scripts/phase38_smoke.sh
```

This validates plumbing and determinism, not a quality or retention claim.
