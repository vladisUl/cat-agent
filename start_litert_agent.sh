#!/bin/bash

set -e

cd /opt/cat-agent

export PYTHONPATH=/opt/cat-agent/src
export LITERT_AGENT_MODEL_PATH="${LITERT_AGENT_MODEL_PATH:-/storage/models/litertlm/gemma-4-E4B-it.litertlm}"
export LITERT_AGENT_SPECULATIVE="${LITERT_AGENT_SPECULATIVE:-0}"

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.main
