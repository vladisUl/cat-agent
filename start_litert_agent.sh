#!/bin/bash

set -euo pipefail

cd /opt/cat-agent
export PYTHONPATH=/opt/cat-agent/src

if [[ $# -ne 0 ]]; then
    echo "LiteRT profile is configured by litert.active_profile in cat-agent.yaml" >&2
    exit 2
fi

eval "$(/opt/litert-lm-venv/bin/python3 -m orchestration.config_env litert)"

# Benchmark skill forcing remains a low-level diagnostic override, not normal
# application configuration.
export LITERT_AGENT_BENCH_SKILLS="${LITERT_AGENT_BENCH_SKILLS:-}"

# YNNPACK reports unsupported delegation candidates as ERROR even though
# LiteRT falls back normally. Suppress only those capability-check messages;
# preserve every other native stderr line.
YNNPACK_NOISE_RE='^ERROR: third_party/tensorflow/lite/delegates/ynnpack/.*(was not true\.|is not supported\.?)$'

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.main \
    2> >(grep --line-buffered -vE -- "$YNNPACK_NOISE_RE" >&2)
