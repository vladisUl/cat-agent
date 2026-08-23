#!/usr/bin/env bash
set -euo pipefail

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

exec /opt/cat-agent-firebase-venv/bin/python3 -m litert_agent.firebase_gateway
