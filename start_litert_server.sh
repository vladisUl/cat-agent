#!/bin/bash
set -e

source /opt/litert-lm-venv/bin/activate

HOST="${LITERT_AGENT_HOST:-127.0.0.1}"
PORT="${LITERT_AGENT_PORT:-9379}"

echo "LiteRT-LM server: ${HOST}:${PORT}"
echo "Model is selected lazily by each OpenAI request."
exec litert-lm serve \
  --host "$HOST" \
  --port "$PORT"
