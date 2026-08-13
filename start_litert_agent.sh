#!/bin/bash

set -e

cd /opt/cat-agent

DEFAULT_MODEL="/storage/models/litertlm/gemma-4-E4B-it.litertlm"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 [MODEL_PATH] [cpu|gpu]"
  echo
  echo "Defaults:"
  echo "  MODEL_PATH=$DEFAULT_MODEL"
  echo "  backend=cpu"
  echo
  echo "Examples:"
  echo "  $0"
  echo "  $0 /storage/models/litertlm/gemma-4-E4B-it.litertlm cpu"
  echo "  $0 /storage/models/litertlm/gemma-4-E2B-it.litertlm gpu"
  exit 0
fi

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [MODEL_PATH] [cpu|gpu]" >&2
  exit 2
fi

MODEL_PATH="${1:-${LITERT_AGENT_MODEL_PATH:-$DEFAULT_MODEL}}"
BACKEND="${2:-${LITERT_AGENT_BACKEND:-cpu}}"
BACKEND="${BACKEND,,}"

case "$BACKEND" in
  cpu|gpu)
    ;;
  *)
    echo "Invalid LiteRT backend: $BACKEND (expected cpu or gpu)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "LiteRT model not found: $MODEL_PATH" >&2
  exit 1
fi

export PYTHONPATH=/opt/cat-agent/src
export LITERT_AGENT_MODEL_PATH="$MODEL_PATH"
export LITERT_AGENT_BACKEND="$BACKEND"
export LITERT_AGENT_SPECULATIVE="${LITERT_AGENT_SPECULATIVE:-0}"

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.main
