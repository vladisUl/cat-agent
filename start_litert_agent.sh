#!/bin/bash
set -e

cd "$(dirname "$0")"

export CAT_AGENT_API_BASE_URL="${CAT_AGENT_API_BASE_URL:-http://127.0.0.1:9379/v1}"
export CAT_AGENT_MODEL="${CAT_AGENT_MODEL:-gemma4-e4b,cpu}"

echo "LiteRT-LM cat-agent model: $CAT_AGENT_MODEL"
echo "LiteRT-LM API: $CAT_AGENT_API_BASE_URL"

exec python3 -m litert_agent.main
