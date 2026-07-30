# Contributing to aarambh-studio

Thank you for taking the time to contribute. Every bug report, feature suggestion, documentation improvement, and pull request makes aarambh-studio better for everyone.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Ways to Contribute](#ways-to-contribute)
3. [Setting Up the Workspace](#setting-up-the-workspace)
4. [Project Structure](#project-structure)
5. [Making a Change](#making-a-change)
6. [Commit Messages](#commit-messages)
7. [Testing Requirements](#testing-requirements)
8. [Documentation Requirements](#documentation-requirements)
9. [Pull Request Process](#pull-request-process)
10. [Reporting Bugs](#reporting-bugs)
11. [Suggesting Features](#suggesting-features)
12. [Working on Custom Kernels](#working-on-custom-kernels)
13. [Crate Versioning](#crate-versioning)

---

## Code of Conduct

This project follows a simple rule: be respectful. Constructive criticism of code and ideas is welcome; personal attacks are not. Contributors who engage in hostile behavior will be asked to stop and may be removed from the project.

---

## Ways to Contribute

You do not need to write code to contribute:

- **Report a bug** — open an issue with a minimal reproduction
- **Suggest a feature** — open an issue describing the use case, not just the API
- **Improve documentation** — fix typos, add examples, clarify confusing sections
- **Write an example** — show aarambh-studio being used in a real scenario
- **Write a benchmark** — help identify performance regressions
- **Review pull requests** — read others' changes and leave thoughtful feedback
- **Write tests** — increase coverage for existing code

---

## Setting Up the Workspace

### Prerequisites

- Rust 1.89 or later (`rustup update stable`)
- No GPU required for development (Tiny model trains on any i3)

### Clone and build

```sh
git clone https://github.com/AarambhDevHub/aarambh-studio.git
cd aarambh-studio

# Build all crates:
cargo build --workspace --locked

# Run all tests:
cargo test --workspace --locked

# Check linting:
cargo clippy --workspace --all-targets --locked -- -D warnings -D clippy::undocumented_unsafe_blocks

# Check formatting:
cargo fmt --all --check
```

### IDE Setup

This is a standard Cargo workspace. Any IDE with `rust-analyzer` support (VS Code, IntelliJ, Neovim) will work out of the box. Open the root `aarambh-studio/` folder.

---

## Project Structure

```
aarambh-studio/
├── Cargo.toml              ← Workspace manifest (pinned deps)
├── crates/
│   ├── aarambh-studio-core/        ← Foundation types — start here if unsure
│   ├── aarambh-studio-tokenizer/   ← BPE tokeniser, vocab, special tokens
│   ├── aarambh-studio-data/        ← Datasets, preprocessing, data loader
│   ├── aarambh-studio-nn/          ← RMSNorm, RoPE, GQA, SwiGLU, TransformerBlock
│   ├── aarambh-studio-kernel/      ← Custom CUDA + CPU SIMD kernels
│   ├── aarambh-studio-model/       ← Embedding, LM head, full model forward pass
│   ├── aarambh-studio-weights/     ← SafeTensors I/O, GGUF, HF conversion
│   ├── aarambh-studio-quant/       ← INT8, GPTQ, AWQ, GGUF, KV cache quant
│   ├── aarambh-studio-train/       ← Training loop, AdamW, cosine schedule
│   ├── aarambh-studio-finetune/    ← LoRA, QLoRA, SFT, GRPO, verifiers
│   ├── aarambh-studio-inference/   ← Inference engine, KV cache, sampler
│   ├── aarambh-studio-safety/      ← Input/output guardrails, PII, audit
│   ├── aarambh-studio-selflearn/   ← Self-learning loop, replay buffer, critique
│   ├── aarambh-studio-eval/        ← Evaluation harness and scorecards
│   ├── aarambh-studio-vision/      ← Frozen vision encoder and projector
│   └── aarambh-studio-serve/       ← OpenAI-compatible local inference server
└── aarambh-studio/             ← CLI binary
```

Each crate is self-contained. If you are working on attention, you should only need to open `crates/aarambh-studio-nn/`. You should not need to understand `aarambh-studio-quant` to fix an attention bug.

---

## Making a Change

### 1. Check for an existing issue

Search [open issues](https://github.com/AarambhDevHub/aarambh-studio/issues) before starting work. If there is no issue for your change, open one first — especially for anything larger than a typo fix. This prevents duplicate work and gives maintainers a chance to give early feedback on direction.

### 2. Fork and branch

```sh
git clone https://github.com/YOUR_USERNAME/aarambh-studio.git
cd aarambh-studio
git checkout -b fix/adamw-beta2-default
```

Branch naming conventions:

| Change type | Prefix | Example |
|-------------|--------|---------|
| Bug fix | `fix/` | `fix/adamw-beta2-default` |
| New feature | `feat/` | `feat/rmsnorm-simd` |
| Documentation | `docs/` | `docs/training-guide` |
| Refactor | `refactor/` | `refactor/kv-cache-layout` |
| Performance | `perf/` | `perf/parallel-attention` |
| Tests | `test/` | `test/gqa-shape-edge-cases` |

### 3. Make the smallest possible change

Do not mix unrelated changes in one PR. A PR that fixes a bug in `RMSNorm` should not also add a new sampler variant. Keep the diff focused — it makes review faster and easier to revert if something goes wrong.

### 4. Reference the roadmap

Each phase of development is documented in [ROADMAP.md](ROADMAP.md). If your change corresponds to a roadmap task, reference the phase in your PR description.

### 5. Format your code

```sh
cargo fmt
```

The CI rejects unformatted code. Run `cargo fmt` before every commit.

---

## Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <short description>
```

**Type** must be one of:

| Type | When to use |
|------|-------------|
| `feat` | New feature or behavior |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or fixing tests |
| `perf` | Performance improvement without behavior change |
| `refactor` | Code restructuring without behavior change |
| `chore` | Build system, CI, dependency updates |
| `ci` | CI configuration changes |

**Scope** is the affected crate (without the `aarambh-studio-` prefix):

```
feat(core): add ModelConfig::from_json
fix(nn): clamp rmsnorm epsilon to positive
docs(train): add training loop example
test(inference): add kvcache seqlen growth test
perf(kernel): simd rmsnorm avx2 path
```

**Rules:**
- Use the imperative mood: "add", "fix", "update" — not "added", "fixed", "updated"
- Keep the first line under 72 characters
- Reference the issue number in the footer: `Closes #42`
- Breaking changes must include `BREAKING CHANGE:` in the footer

---

## Testing Requirements

Every pull request must include tests. There are no exceptions.

### What to test

- **New behavior:** Write a test that fails before your change and passes after.
- **Bug fixes:** Write a test that reproduces the bug, then fix it.
- **Edge cases:** Empty sequences, zero-length tensors, boundary dimensions, single-token inputs.

### Where to put tests

- Small unit tests go in a `#[cfg(test)]` block at the bottom of the relevant `src/*.rs` file.
- Integration tests that span multiple modules go in `tests/` at the crate root or workspace root.

### Running tests

```sh
# Unit + integration tests for all crates:
cargo test --workspace

# A specific crate only:
cargo test -p aarambh-studio-core

# A specific test by name:
cargo test -p aarambh-studio-core tiny_config_head_dim_is_correct
```

### Clippy

The CI runs:

```sh
cargo clippy --workspace --all-targets -- -D warnings
```

This means any clippy warning is a CI failure. Fix all warnings before pushing.

---

## Documentation Requirements

- Every `pub` item (struct, enum, trait, function, method) must have a `///` doc comment.
- Doc comments must include at least a one-sentence description.
- Non-trivial APIs must include a `# Examples` section with a runnable code block.
- Doc examples are compiled and run by `cargo test --doc` — they must work.

Check that your docs render correctly:

```sh
cargo doc --workspace --open
```

---

## Pull Request Process

### Before opening

Run this checklist locally:

```sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

All must pass cleanly — zero warnings, zero failures.

### Opening the PR

- Fill in the pull request template fully.
- Link the related issue: "Closes #42" or "Related to #42".
- Keep the title in Conventional Commits format: `fix(nn): clamp rmsnorm epsilon to prevent div by zero`.
- If the PR is a work in progress, open it as a Draft.

### Review process

- At least one maintainer approval is required before merging.
- Address all review comments — if you disagree with feedback, explain your reasoning in the thread. Do not silently ignore it.
- Keep the PR up to date with `main` by rebasing, not merging.
- Once approved, the maintainer will squash-merge the PR.

### After merging

Your change will appear in the next release. If it is a fix or small feature, it will go in the next patch or minor release. Large features wait for the next planned milestone.

---

## Reporting Bugs

Open an issue using the **Bug Report** template. Include:

1. **What you expected to happen.**
2. **What actually happened** — include the full error message or unexpected output.
3. **A minimal reproduction** — the smallest possible code that demonstrates the bug. Remove everything unrelated.
4. **Environment** — Rust version (`rustc --version`), OS, aarambh-studio version, active feature flags.

A minimal reproduction is the single most important thing you can provide. Issues without one may be closed if the bug cannot be reproduced.

---

## Suggesting Features

Open an issue using the **Feature Request** template. Include:

1. **The use case** — what are you trying to accomplish? Why does the current API not solve it?
2. **Proposed API** — what would you want to write? Show code.
3. **Alternatives considered** — what workarounds exist today and why are they insufficient?

Feature requests that describe only the desired API without explaining the use case will be asked for more context before being accepted.

---

## Working on Custom Kernels

The `aarambh-studio-kernel` crate owns CUDA C and raw SIMD pointer arithmetic.
`aarambh-studio-weights` and `aarambh-studio-vision` each contain one audited
memory-mapped SafeTensors boundary. Every unsafe block requires a `SAFETY:`
explanation and must pass `clippy::undocumented_unsafe_blocks`.

### CPU SIMD

CPU SIMD kernels use stable `std::arch` intrinsics with runtime dispatch for
AVX2/FMA, AVX512, AVX2, and scalar fallback paths. No nightly toolchain is
required.

### CUDA kernels

CUDA kernels require NVCC. If not found at build time, the crate falls back to candle:

```sh
# Verify NVCC is available
which nvcc
```

### Testing kernel fallback

```sh
cargo test -p aarambh-studio-kernel
```

All tests should pass even without NVCC — they test the candle fallback path.

---

## Crate Versioning

aarambh-studio follows [Semantic Versioning](https://semver.org/).

- **Patch** (`2.0.x`) — bug fixes only, no API changes.
- **Minor** (`2.x.0`) — new features and backward-compatible API additions.
- **Major** (`x.0.0`) — breaking API changes with migration notes.

All sub-crates share the workspace version and remain internal, non-publishable
implementation units. Aarambh AI is distributed from GitHub as an application.

---

## Questions?

If you are unsure about anything — whether a bug is worth reporting, whether a feature fits the project, or how to approach a change — open an issue and ask.
