#!/bin/bash

set -e

: "${CAT_AGENT_MODEL:?CAT_AGENT_MODEL is not set}"

cd /opt/cat-agent

export PYTHONPATH=/opt/cat-agent/src
export CAT_AGENT_API_BASE_URL=http://127.0.0.1:9380/v1

exec python3 -m llama_agent.main
