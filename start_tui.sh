#!/bin/bash

set -e

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.terminal
