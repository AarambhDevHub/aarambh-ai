# Phase 31 MoE Sweep

This guide defines the reproducible comparison for Phase 31 fine-grained MoE.
It does not contain fabricated Medium or Large scores. Running the sweep writes
the measured scorecards and comparisons to an untracked output directory.

## Configuration Contract

The split factor changes expert granularity without changing total routed FFN
capacity:

```text
routed_experts = num_experts * fine_grained_factor
fine_width = expert_ffn_dim / fine_grained_factor
routed_capacity_width = routed_experts * fine_width
active_routed_width = top_k * fine_width
```

For a matched active-compute comparison, multiply the coarse `top_k` by the
split factor. Shared experts are separate always-on capacity and must be
reported separately rather than hidden inside the routed budget.

The shipped Medium target uses 32 routed experts (`8 × 4`), top-k 8, fine width
424, and one shared expert. The Large target uses the same pool and routing
shape with fine width 832. Their corresponding coarse configs use 8 routed
experts and top-k 2.

## Warm Start

Train the corresponding coarse config first, then launch the fine-grained
target config. The target's `[moe_retrofit] source_top_k = 2` contract requires
target top-k 8 for factor 4. Loading performs these transformations:

1. Repeat each coarse router row once for every child expert.
2. Partition `w_gate` and `w_up` output rows by child.
3. Partition `w_down` input columns and multiply each child projection by the
   split factor.
4. Keep new shared gate/up initialization and zero the shared down projection.

The repeated router gives sibling experts equal initial probability. Scaling
the child down projections compensates for the probability divided among those
siblings, preserving the source MoE output within floating-point tolerance.
The zero shared down projection makes the new path initially neutral.

## Small Sweep

Build once, then use a scratch-training config containing `[model.moe]`:

```sh
cargo build --release -p aarambh-studio --locked
scripts/phase31_sweep_moe.sh \
  configs/moe_smoke.toml \
  data/eval \
  runs/phase31_moe_sweep
```

Defaults evaluate perplexity for four cases: coarse 8×1, fine 8×2, fine 8×4,
and wide 16×4. The first case is the comparison baseline. Override the run
without editing the script:

```sh
TASKS=ppl,gsm8k-subset \
MAX_EXAMPLES=256 \
MAX_STEPS=1000 \
CASES="coarse:8:1:2:0 fine:8:4:8:1" \
scripts/phase31_sweep_moe.sh \
  configs/moe_smoke.toml data/eval runs/phase31_moe_sweep
```

Each `CASES` entry is
`name:num_experts:fine_grained_factor:top_k:num_shared_experts`. The base
expert width must divide by every requested factor. Retrofit configs are
rejected because mixing warm-start histories would invalidate a scratch sweep.

## Outputs

The output directory contains generated TOML files, final checkpoints,
training logs, JSON scorecards, per-case Markdown scorecards, baseline-relative
comparisons, and `results.md`. Keep the following fixed when interpreting
deltas:

- tokenizer, data split, random seed, optimizer, update count, and sequence
  length;
- coarse expert width before subdivision;
- routed active width for matched-compute cases;
- evaluation examples and generation budget.

Report total parameters, routed active parameters, shared active parameters,
tokens per second, peak memory, expert utilization range, and dead experts
alongside quality metrics. Dense dispatch computes every routed expert in this
release, so conceptual active parameters do not imply proportional wall-clock
savings; the sweep measures quality and specialization under the current
execution contract.
