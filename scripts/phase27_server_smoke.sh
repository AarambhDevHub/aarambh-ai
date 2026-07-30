#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/tiny_shakespeare.toml}"
MODEL="${MODEL:-checkpoints/tiny_shakespeare/step_000050/model.safetensors}"
TOKENIZER="${TOKENIZER:-checkpoints/tiny_shakespeare/tokenizer.json}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
MODEL_ID="${MODEL_ID:-aarambh-studio-smoke}"
BASE_URL="http://${HOST}:${PORT}"

if [[ ! -x target/release/aarambh-studio ]]; then
  cargo build --release -p aarambh-studio
fi

target/release/aarambh-studio serve \
  --config "$CONFIG" \
  --model "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --model-id "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --max-batch-size 4 \
  --queue-capacity 16 \
  --safety strict &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 100); do
  if curl --fail --silent "$BASE_URL/readyz" >/dev/null; then
    break
  fi
  sleep 0.1
done
curl --fail --silent "$BASE_URL/readyz" >/dev/null

curl --fail --silent "$BASE_URL/v1/models" | grep -q "$MODEL_ID"

curl --fail --silent \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":4,\"temperature\":0}" \
  "$BASE_URL/v1/chat/completions" | grep -q 'chat.completion'

curl --fail --silent --no-buffer \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":4,\"temperature\":0,\"stream\":true}" \
  "$BASE_URL/v1/chat/completions" | grep -q '\[DONE\]'

pids=()
for prompt in One Two Three Four; do
  curl --fail --silent \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"$prompt\",\"max_tokens\":4,\"temperature\":0}" \
    "$BASE_URL/v1/completions" >/dev/null &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

curl --fail --silent "$BASE_URL/metrics" | grep -q 'decode_batches'
echo "Phase 27 server smoke passed at $BASE_URL"
