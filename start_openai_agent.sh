#!/bin/bash

set -euo pipefail

cd /opt/cat-agent

ENV_FILE="${CAT_AGENT_OPENAI_ENV_FILE:-/opt/cat-agent/.env.openai}"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

export PYTHONPATH=/opt/cat-agent/src

# Ordinary OpenAI settings come from cat-agent.yaml. The local env file is
# intentionally reserved for secrets such as API keys.
CONFIG_ENV="$(/opt/litert-lm-venv/bin/python3 -m orchestration.config_env openai)"
eval "$CONFIG_ENV"
unset CONFIG_ENV

exec /opt/litert-lm-venv/bin/python3 -m openai_agent.main
