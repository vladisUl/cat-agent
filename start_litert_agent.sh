#!/bin/bash
set -e

cd "$(dirname "$0")"

# Keep LiteRT-LM transport selection isolated from llama.cpp environment.
# CAT_AGENT_MODEL is commonly exported by the llama launcher as a GGUF path;
# never inherit that value here.
export CAT_AGENT_API_BASE_URL="${LITERT_AGENT_API_BASE_URL:-http://127.0.0.1:9379/v1}"
export CAT_AGENT_MODEL="${LITERT_AGENT_MODEL:-gemma4-e4b,cpu}"

echo "LiteRT-LM cat-agent model: $CAT_AGENT_MODEL"
echo "LiteRT-LM API: $CAT_AGENT_API_BASE_URL"

exec python3 -m litert_agent.main
