## Description

Please include a summary of the change and which issue is fixed.

Fixes #(issue)

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist

- [ ] `cargo check --workspace --all-targets --locked` passes
- [ ] `cargo test --workspace --no-fail-fast --locked` passes
- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --workspace --all-targets --locked -- -D warnings -D clippy::undocumented_unsafe_blocks` passes
- [ ] Public API documentation passes `-D missing_docs`
- [ ] No TODOs, placeholders, generated weights, or model artifacts were added
- [ ] Corresponding tests added (if applicable)
- [ ] CHANGELOG.md updated (if applicable)
- [ ] ROADMAP.md phase checklist updated (if applicable)
