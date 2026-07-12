#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_VERSION="2.0.0"
EXPECTED_PACKAGES=17

command -v jq >/dev/null 2>&1 || {
  echo "release audit requires jq" >&2
  exit 1
}

[[ -f Cargo.lock ]] || {
  echo "Cargo.lock is required for the v2 application release" >&2
  exit 1
}

if git check-ignore --quiet Cargo.lock; then
  echo "Cargo.lock must not be ignored" >&2
  exit 1
fi

metadata="$(cargo metadata --locked --no-deps --format-version 1)"
package_count="$(jq '.packages | length' <<<"$metadata")"
if [[ "$package_count" -ne "$EXPECTED_PACKAGES" ]]; then
  echo "expected $EXPECTED_PACKAGES workspace packages, found $package_count" >&2
  exit 1
fi

invalid_packages="$(
  jq -r --arg version "$EXPECTED_VERSION" '
    .packages[]
    | select(.version != $version or (.publish | length) != 0)
    | "\(.name): version=\(.version) publish=\(.publish)"
  ' <<<"$metadata"
)"
if [[ -n "$invalid_packages" ]]; then
  echo "all packages must be version $EXPECTED_VERSION and publish=false:" >&2
  echo "$invalid_packages" >&2
  exit 1
fi

cli_version="$(cargo run --quiet --locked -p aarambh-ai -- --version)"
if [[ "$cli_version" != "aarambh-ai $EXPECTED_VERSION" ]]; then
  echo "unexpected CLI version: $cli_version" >&2
  exit 1
fi

if git ls-files '*.rs' '*.cu' '*.cuh' '*.h' \
  | xargs -r rg -n 'TODO|FIXME|HACK|XXX|todo!\s*\(|unimplemented!\s*\(|not implemented';
then
  echo "unfinished implementation markers are not allowed in release sources" >&2
  exit 1
fi

if rg -n -U '__global__[^\{]+\{\s*\}' crates/aarambh-ai-kernel/kernels -g '*.cu'; then
  echo "empty CUDA kernel bodies are not allowed" >&2
  exit 1
fi

if rg -n '^\s*\[ \]' ROADMAP.md ROADMAP_V2.md; then
  echo "roadmap checklists must not contain unfinished tasks" >&2
  exit 1
fi

tracked_artifacts="$(
  git ls-files \
    '*.safetensors' '*.gguf' '*.ckpt' '*.pt' '*.pth' '*.onnx' \
    'checkpoints/**' 'adapters/**'
)"
if [[ -n "$tracked_artifacts" ]]; then
  echo "model artifacts must not be tracked in the source release:" >&2
  echo "$tracked_artifacts" >&2
  exit 1
fi

if rg -n 'cargo publish' .github/workflows; then
  echo "GitHub workflows must not publish crates" >&2
  exit 1
fi

echo "Phase 28 release audit passed for aarambh-ai $EXPECTED_VERSION"
