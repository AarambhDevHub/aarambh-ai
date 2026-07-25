# v3.0.0 Release Runbook

aarambh-ai v3.0.0 is a GitHub application source release. All workspace
packages remain `publish = false`. crates.io publishing is deferred to v4.0.0.
Do not publish crates, upload compiled binaries, or attach pretrained
checkpoints, adapters, tokenizers, optimizer state, SafeTensors, or GGUF files.

## Release Requirements

- Rust 1.89 or newer, `jq`, and the committed `Cargo.lock`.
- A clean `main` branch containing the reviewed Phase 40 release commit.
- Green stable, MSRV, RustSec, documentation, test, and release-audit checks.
- Optional CUDA runtime evidence recorded from a CUDA/NVCC host; CPU fallback
  remains the portable release baseline.

## Validate The Source

```sh
cargo fmt --all --check
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- \
  -D warnings -D clippy::undocumented_unsafe_blocks
cargo test --workspace --no-fail-fast --locked
RUSTDOCFLAGS="-D warnings -D missing_docs" \
  cargo doc --workspace --no-deps --locked
cargo audit
scripts/phase40_release_audit.sh
cargo build --release -p aarambh-ai --locked
```

RustSec maintenance warnings without a known vulnerability are reviewed as
dependency status, not silently ignored. A known vulnerability blocks the
release.

## Validate The CLI

```sh
test "$(target/release/aarambh-ai --version)" = "aarambh-ai 3.0.0"
target/release/aarambh-ai --help
target/release/aarambh-ai train --help
target/release/aarambh-ai infer --help
target/release/aarambh-ai agent --help
target/release/aarambh-ai eval --help
target/release/aarambh-ai quantise --help
target/release/aarambh-ai convert --help
target/release/aarambh-ai finetune --help
target/release/aarambh-ai distill --help
target/release/aarambh-ai selflearn --help
target/release/aarambh-ai serve --help
```

Verify a clean source installation without changing the user Cargo home:

```sh
rm -rf /tmp/aarambh-ai-v3-install
cargo install --path aarambh-ai --locked --root /tmp/aarambh-ai-v3-install
/tmp/aarambh-ai-v3-install/bin/aarambh-ai --version
```

When a local Tiny checkpoint is available, run the inference-server smoke test:

```sh
scripts/phase27_server_smoke.sh
```

## Optional CUDA Validation

Run this on Kaggle or another CUDA/NVCC host before tagging when GPU access is
available:

```sh
cargo check --workspace --all-targets --features cuda --locked
cargo test -p aarambh-ai-kernel --features cuda --locked
cargo run --release --locked -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_cuda_smoke.toml
```

The A100 speed targets are hardware measurements, not correctness gates. CUDA
numerical tests and CPU/Candle fallback behavior remain release requirements.

## Tag And Publish

After the Phase 40 pull request is merged and `main` is current:

```sh
git switch main
git pull --ff-only origin main
scripts/phase40_release_audit.sh
git tag -a v3.0.0 -m "aarambh-ai v3.0.0"
git push origin v3.0.0
```

The tag triggers `.github/workflows/release.yml`. Verify that the resulting
GitHub Release:

- is named `aarambh-ai v3.0.0` and marked latest;
- uses `.github/release-notes/v3.0.0.md`;
- points at the intended `main` commit;
- contains only GitHub's automatic source archives;
- contains no uploaded binary or model artifacts.

## Prohibited Release Actions

- Do not run `cargo publish` or add a crates.io token.
- Do not upload checkpoints, adapters, tokenizers, optimizer state, or GGUF.
- Do not upload prebuilt CPU or CUDA binaries for v3.0.0.
- Do not tag from an unreviewed branch or with a dirty working tree.
