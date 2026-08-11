#!/bin/bash

set -e

: "${CAT_AGENT_MODEL:?CAT_AGENT_MODEL is not set}"

MTP_MODEL=/opt/llama.cpp/models/gemma-4/mtp-gemma-4-E4B-it.gguf

if [[ ! -f "$MTP_MODEL" ]]; then
  echo "MTP model not found: $MTP_MODEL" >&2
  exit 1
fi

echo "Model: $CAT_AGENT_MODEL"
echo "MTP: $MTP_MODEL (draft-mtp, n_max=3, CPU)"
echo "Slots: 2, ctx-checkpoints=0, cache-prompt=on, threads-batch=4"

cd /opt/llama.cpp

exec ./build-vulkan/bin/llama-server \
  -m "$CAT_AGENT_MODEL" \
  --host 0.0.0.0 \
  --port 9380 \
  --parallel 2 \
  --device none \
  --threads-batch 4 \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --cache-prompt \
  --cache-ram 0 \
  --ctx-checkpoints 0 \
  --spec-type draft-mtp \
  --spec-draft-model "$MTP_MODEL" \
  --spec-draft-device none \
  --spec-draft-ngl 0 \
  --spec-draft-n-max 3
