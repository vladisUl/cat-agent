#!/bin/bash

set -e

: "${CAT_AGENT_MODEL:?CAT_AGENT_MODEL is not set}"

cd /opt/llama.cpp

extra_args=()
case "$(basename "$CAT_AGENT_MODEL")" in
  gemma-4-E4B-it-*.gguf)
    mtp_model=/opt/llama.cpp/models/gemma-4/mtp-gemma-4-E4B-it.gguf
    if [[ ! -f "$mtp_model" ]]; then
      echo "MTP model not found: $mtp_model" >&2
      exit 1
    fi
    extra_args+=(
      --spec-type draft-mtp
      --spec-draft-model "$mtp_model"
      --spec-draft-device none
      --spec-draft-ngl 0
      --spec-draft-n-max 3
    )
    ;;
esac

exec ./build-vulkan/bin/llama-server \
  -m "$CAT_AGENT_MODEL" \
  --host 0.0.0.0 \
  --port 9380 \
  --parallel 2 \
  --device none \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --cache-prompt \
  --cache-ram 0 \
  --ctx-checkpoints 0 \
  "${extra_args[@]}"
