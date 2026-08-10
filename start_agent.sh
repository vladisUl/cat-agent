#!/bin/bash

set -e

cd /opt/cat-agent

export CAT_AGENT_API_BASE_URL=http://127.0.0.1:9380/v1
export CAT_AGENT_MODEL='/opt/llama.cpp/models/gemma-4/gemma-4-E4B-it-Q4_K_M.gguf'

exec python3 -m cat_agent.main
