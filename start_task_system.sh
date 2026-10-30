#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${CAT_AGENT_TASK_PYTHON:-/opt/litert-lm-venv/bin/python3}"

# No arguments run the long-lived service; arguments use its administrative
# CLI. Both share the same Task SYSTEM socket and configuration.
if (( $# == 0 )); then
    exec "$PYTHON" -m task_system.main
fi
exec "$PYTHON" -m task_system.cli "$@"
