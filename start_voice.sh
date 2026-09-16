#!/bin/bash

set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/select_core.sh" "$@"

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

CONFIG_ENV="$(/opt/litert-lm-venv/bin/python3 -m orchestration.config_env voice)"
eval "$CONFIG_ENV"
unset CONFIG_ENV

exec /opt/gigaam/env/bin/python3 -m litert_agent.voice_entry
