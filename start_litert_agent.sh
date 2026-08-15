#!/bin/bash

set -e

cd /opt/cat-agent

MODE="${1:-cpu}"
PROFILE="${2:-normal}"

case "$MODE" in
    cpu)
        MODEL_PATH="/storage/models/litertlm/gemma-4-E4B-it.litertlm"
        BACKEND="cpu"
        ACTIVATION_DATA_TYPE=""
        ENABLE_THINKING=""
        ;;
    gpu)
        MODEL_PATH="/storage/models/litertlm/gemma-4-E4B-it-gpu.litertlm"
        BACKEND="gpu"
        ACTIVATION_DATA_TYPE="fp32"
        ENABLE_THINKING=""
        ;;
    qwen)
        MODEL_PATH="/storage/models/litertlm/qwen3_4b_mixed_int4.litertlm"
        BACKEND="cpu"
        ACTIVATION_DATA_TYPE=""
        ENABLE_THINKING="0"
        ;;
    *)
        echo "Usage: $0 [cpu|gpu|qwen] [normal|bench]" >&2
        exit 2
        ;;
esac

case "$PROFILE" in
    normal)
        BENCH_SKILLS=""
        ;;
    bench)
        BENCH_SKILLS="mqtt,shell"
        ;;
    *)
        echo "Usage: $0 [cpu|gpu|qwen] [normal|bench]" >&2
        exit 2
        ;;
esac

export PYTHONPATH=/opt/cat-agent/src
export LITERT_AGENT_MODEL_PATH="$MODEL_PATH"
export LITERT_AGENT_BACKEND="$BACKEND"
export LITERT_AGENT_ACTIVATION_DATA_TYPE="$ACTIVATION_DATA_TYPE"
export LITERT_AGENT_SPECULATIVE="0"
export LITERT_AGENT_BENCH_SKILLS="$BENCH_SKILLS"
export LITERT_AGENT_ENABLE_THINKING="$ENABLE_THINKING"

exec /opt/litert-lm-venv/bin/python3 -m litert_agent.main
