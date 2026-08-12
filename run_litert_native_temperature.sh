#!/bin/bash
set -e

cd "$(dirname "$0")"
source /opt/litert-lm-prefill-venv/bin/activate

export CAT_AGENT_MODEL="litert-native-e4b"
export LITERT_AGENT_MODEL_PATH="${LITERT_AGENT_MODEL_PATH:-/opt/litert-lm/models/gemma-4-E4B-it/gemma-4-E4B-it.litertlm}"
export LITERT_AGENT_BACKEND="${LITERT_AGENT_BACKEND:-cpu}"
export LITERT_AGENT_SPECULATIVE="${LITERT_AGENT_SPECULATIVE:-0}"

exec python3 -m litert_agent.native_temperature_shot
