# Aarambh AI Inference Server

Phase 27 exposes one local checkpoint through an OpenAI-compatible HTTP/SSE
API. The server does not download, publish, or execute model-generated tool
calls.

## Start The Server

```bash
cargo run --release -p aarambh-studio -- serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --tokenizer checkpoints/tiny_shakespeare/tokenizer.json \
  --model-id aarambh-tiny \
  --port 8080
```

The default bind is `127.0.0.1:8080`. For a non-loopback bind, set a key before
starting the server:

```bash
export AARAMBH_STUDIO_STUDIO_API_KEY='replace-with-a-long-random-secret'
cargo run --release -p aarambh-studio -- serve \
  --config configs/tiny_shakespeare.toml \
  --model checkpoints/tiny_shakespeare/step_000050/model.safetensors \
  --model-id aarambh-tiny \
  --host 0.0.0.0
```

## Chat Completion

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "aarambh-tiny",
    "messages": [{"role": "user", "content": "To be, or not to be"}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Set `"stream": true` to receive `chat.completion.chunk` SSE events followed by
`data: [DONE]`. With safety enabled, generated fragments pass through a rolling
cross-token scanner before they are released. PII is redacted and toxic output
terminates with `finish_reason: "content_filter"`.

## OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
response = client.chat.completions.create(
    model="aarambh-tiny",
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=32,
)
print(response.choices[0].message.content)
```

When local authentication is disabled, SDKs may still require a non-empty
client-side `api_key`; any placeholder is accepted because the local server does
not validate authorization unless `AARAMBH_STUDIO_STUDIO_API_KEY` is configured.

## Endpoints

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

Phase 27 supports one text model and one generated choice per request. Vision,
self-learning, speculative server decoding, parallel tool calls, tool-result
history, and `/v1/responses` are not part of this phase.

## Smoke Test

```bash
bash scripts/phase27_server_smoke.sh
```

Override `CONFIG`, `MODEL`, `TOKENIZER`, `HOST`, `PORT`, or `MODEL_ID` when the
checkpoint layout differs.
