# Release Runbook

This project uses GitHub source releases for v1.0.0. Do not publish crates to crates.io and do not attach pretrained checkpoints or model artifacts for this release line.

## v1.0.0 Checklist

1. Start from a clean `main` branch.
2. Confirm all crate manifests use `version = "1.0.0"` and `publish = false`.
3. Confirm `README.md`, `ROADMAP.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `SECURITY.md`, `CONTRIBUTING.md`, and `.github/release-notes/v1.0.0.md` describe the source-release policy.
4. Run CPU/default validation:

```sh
cargo fmt --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo build --release -p aarambh-ai
```

5. Run CLI smoke checks:

```sh
cargo run -p aarambh-ai -- --version
cargo run -p aarambh-ai -- --help
cargo run -p aarambh-ai -- train --help
cargo run -p aarambh-ai -- infer --help
cargo run -p aarambh-ai -- quantise --help
cargo run -p aarambh-ai -- convert --help
cargo run -p aarambh-ai -- finetune --help
cargo run -p aarambh-ai -- selflearn --help
```

6. Create and push the release tag:

```sh
git tag v1.0.0
git push origin v1.0.0
```

7. Verify the GitHub Release was created from `.github/release-notes/v1.0.0.md`.

## Do Not Do For v1.0.0

- Do not run `cargo publish`.
- Do not upload pretrained SafeTensors, GGUF, adapter, optimizer, or tokenizer artifacts as release assets.
- Do not attach binary builds unless the release policy changes in a future release.

## Optional CUDA Validation

CUDA validation is manual and must be run on a host with CUDA and NVCC:

```sh
cargo test -p aarambh-ai-kernel --features cuda
cargo run --release -p aarambh-ai --features cuda -- train \
  --config configs/wikitext103_cuda_smoke.toml
```

If NVCC is missing, the build should fall back to CPU/Candle paths and emit only the expected kernel warning.
