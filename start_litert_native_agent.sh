#!/bin/bash
set -e

cd "$(dirname "$0")"
source /opt/litert-lm-prefill-venv/bin/activate

# Keep LiteRT-LM configuration isolated from the llama.cpp environment.
# This venv contains the official 0.14 API plus the local
# prefill_preface_on_init binding patch.
export CAT_AGENT_MODEL="litert-native-e4b"
export LITERT_AGENT_MODEL_PATH="${LITERT_AGENT_MODEL_PATH:-/opt/litert-lm/models/gemma-4-E4B-it/gemma-4-E4B-it.litertlm}"
export LITERT_AGENT_BACKEND="${LITERT_AGENT_BACKEND:-cpu}"
export LITERT_AGENT_SPECULATIVE="${LITERT_AGENT_SPECULATIVE:-0}"

echo "LiteRT-LM native venv: /opt/litert-lm-prefill-venv"
echo "LiteRT-LM native model: $LITERT_AGENT_MODEL_PATH"
echo "LiteRT-LM native backend: $LITERT_AGENT_BACKEND"
echo "LiteRT-LM native speculative: $LITERT_AGENT_SPECULATIVE"

exec python3 -m litert_agent.native_main
