# DoRA vs LoRA Evaluation Notes

Phase 18 adds DoRA/QDoRA as an alternative to LoRA/QLoRA. The repository does
not ship pretrained checkpoints or adapters, so quality comparison must be run
locally from user-created checkpoints.

## Train Matching Adapters

Use the same base checkpoint, tokenizer, SFT dataset, rank, target modules, and
training schedule for both methods.

```sh
cargo run --release -p aarambh-studio -- finetune sft \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --output adapters/tiny_lora \
  --lora-rank 16

cargo run --release -p aarambh-studio -- finetune dora \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --data data/instruct_tiny.jsonl \
  --output adapters/tiny_dora \
  --lora-rank 16
```

## Merge And Evaluate

```sh
cargo run --release -p aarambh-studio -- finetune merge \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --adapter adapters/tiny_lora \
  --method auto \
  --output checkpoints/tiny_lora_merged

cargo run --release -p aarambh-studio -- finetune merge \
  --config configs/tiny_shakespeare.toml \
  --base checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --adapter adapters/tiny_dora \
  --method auto \
  --output checkpoints/tiny_dora_merged

cargo run --release -p aarambh-studio -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_lora_merged/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks ppl,mmlu,hellaswag,gsm8k \
  --data-dir data/eval \
  --out scorecard_lora.json \
  --markdown scorecard_lora.md

cargo run --release -p aarambh-studio -- eval \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_dora_merged/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --tasks ppl,mmlu,hellaswag,gsm8k \
  --data-dir data/eval \
  --out scorecard_dora.json \
  --markdown scorecard_dora.md

cargo run --release -p aarambh-studio -- eval \
  --compare scorecard_lora.json scorecard_dora.json \
  --markdown compare_dora_vs_lora.md
```

HumanEval-lite can be added with `--tasks all --allow-code-exec` when Python
code execution is acceptable in the local environment.
