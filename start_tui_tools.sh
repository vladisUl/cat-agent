#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${CAT_AGENT_TOOLS_PYTHON:-/opt/litert/litert_env/bin/python3}"
exec "$PYTHON" -m agent_core_tools.native_terminal "$@"
