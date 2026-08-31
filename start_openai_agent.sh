#!/bin/bash

set -e

cd /opt/cat-agent

ENV_FILE="${CAT_AGENT_OPENAI_ENV_FILE:-/opt/cat-agent/.env.openai}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

: "${CAT_AGENT_API_BASE_URL:?CAT_AGENT_API_BASE_URL is not set}"
: "${CAT_AGENT_MODEL:?CAT_AGENT_MODEL is not set}"

export PYTHONPATH=/opt/cat-agent/src

exec /opt/litert-lm-venv/bin/python3 -m openai_agent.main
