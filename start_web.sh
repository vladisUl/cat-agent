#!/usr/bin/env bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/select_core.sh" "$@"

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

# This deployment intentionally exposes the Web UI on the trusted home
# network and through the configured HTTPS reverse proxy.
export CAT_AGENT_WEB_TRUSTED_NETWORK=1
export CAT_AGENT_WEB_ORIGINS='https://vlad777.ddns.net,http://192.168.0.140:8080'

# Web telemetry needs the Ollama cloud usage key even when the selected chat
# CORE is LiteRT. Keep the secret server-side; the browser receives only usage.
if [[ -f /opt/cat-agent/.env.openai ]]; then
    set -a
    source /opt/cat-agent/.env.openai
    set +a
fi

CONFIG_ENV="$(/opt/litert-lm-venv/bin/python3 -m orchestration.config_env web)"
eval "$CONFIG_ENV"
unset CONFIG_ENV

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.web_gateway
