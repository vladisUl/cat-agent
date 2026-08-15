#!/bin/bash

set -e

cd /opt/cat-agent

MODEL_PATH="${LITERT_AGENT_MODEL_PATH:-/storage/models/litertlm/gemma-4-E4B-it.litertlm}"
BACKEND="${LITERT_AGENT_BACKEND:-cpu}"
ACTIVATION_DATA_TYPE="${LITERT_AGENT_ACTIVATION_DATA_TYPE:-}"
SPECULATIVE="${LITERT_AGENT_SPECULATIVE:-0}"

export PYTHONPATH=/opt/cat-agent/src
export LITERT_AGENT_MODEL_PATH="$MODEL_PATH"
export LITERT_AGENT_BACKEND="$BACKEND"
export LITERT_AGENT_ACTIVATION_DATA_TYPE="$ACTIVATION_DATA_TYPE"
export LITERT_AGENT_SPECULATIVE="$SPECULATIVE"

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.main
