#!/usr/bin/env bash
set -euo pipefail

source /opt/litert-lm-venv/bin/activate
cd /opt/cat-agent

export PYTHONPATH=/opt/cat-agent
export LITERT_AGENT_MODEL_PATH="${LITERT_AGENT_MODEL_PATH:-/opt/litert-lm/models/gemma-4-E4B-it/gemma-4-E4B-it.litertlm}"
export LITERT_AGENT_BACKEND="${LITERT_AGENT_BACKEND:-cpu}"
export LITERT_AGENT_SPECULATIVE="${LITERT_AGENT_SPECULATIVE:-0}"

exec python3 -m litert_agent.native_015_temperature_shot
