#!/bin/bash

set -e

source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/select_core.sh" "$@"

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

exec /opt/gigaam/env/bin/python3 -m litert_agent.voice
