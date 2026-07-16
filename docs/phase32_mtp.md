# Phase 32 Multi-Token Prediction

Phase 32 adds optional future-token heads to the decoder trunk. It does not
claim quality or throughput gains without measured hardware runs. The checked-in
scripts produce the evidence needed for those claims.

## Configuration

`num_future_tokens` includes the main next-token prediction. The default MTP-2
shape therefore adds one auxiliary head for `t+2`; MTP-3 adds heads for `t+2`
and `t+3`.

```toml
[model.mtp]
num_future_tokens = 2
aux_loss_weight = 0.3
```

`mtp` is optional. Omitting the table preserves the pre-Phase-32 model,
checkpoint, loss, and ordinary inference path. Each auxiliary head has its own
normalization and refinement block, while its vocabulary projection shares the
main LM-head tensor.

## Training And Retrofit

Run the local two-step smoke:

```sh
cargo run --release -p aarambh-ai -- train --config configs/mtp_smoke.toml
```

Training logs report the main loss, the unweighted mean auxiliary loss, its
configured weight, and each
future offset separately. Heads execute sequentially so their vocabulary logits
are consumed immediately into scalar losses rather than collected in a logits
array.

`configs/medium_mtp.toml` and `configs/large_mtp.toml` retrofit the completed
Phase 31 checkpoints. The loader fresh-initializes a complete absent MTP tensor
set and rejects a partially present set. Use a reduced continuation learning
rate because only the new heads begin untrained.

For a matched scratch-training comparison, keep tokenizer, data split, model
trunk, optimizer, seed, sequence length, update count, and evaluation examples
identical:

```sh
scripts/phase32_compare_training.sh \
  configs/tiny_shakespeare_smoke.toml \
  configs/mtp_smoke.toml \
  data/eval \
  runs/phase32_training
```

Do not interpret a two-step smoke run as evidence of sample-efficiency gains.

## One-Checkpoint Speculation

An MTP checkpoint needs no separate draft model:

```sh
target/release/aarambh-ai infer \
  --config configs/mtp_smoke.toml \
  --model checkpoints/mtp_smoke/step_000002/model.safetensors \
  --tokenizer checkpoints/mtp_smoke/tokenizer.json \
  --prompt "To be, or not to be" \
  --max-tokens 64 \
  --greedy \
  --speculative \
  --stats
```

`--draft-tokens` may reduce the proposal width but cannot exceed the trained
horizon. Supplying `--draft-model` and `--draft-config` selects the existing
two-checkpoint draft path instead.

The target verifies each proposal group in one trunk forward and uses the same
modified rejection sampling as external speculative decoding. Greedy output is
required to match ordinary target decoding exactly. Sampled output preserves
the target distribution rather than requiring token-for-token equality.

Benchmark repeated exact greedy output and report source-specific counters:

```sh
scripts/phase32_benchmark_mtp.sh \
  configs/medium_mtp.toml \
  checkpoints/wikitext103_medium_mtp/best/model.safetensors \
  checkpoints/wikitext103_medium/tokenizer.json
```

The output includes target verification forwards, MTP-head forwards, proposal
acceptance, elapsed time, and tokens per second. Report warm and cold results,
hardware, dtype, prompt length, generation length, and acceptance rate with any
throughput comparison.
