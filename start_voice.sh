#!/bin/bash

set -e

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

exec /opt/gigaam/env/bin/python3 -m litert_agent.voice
