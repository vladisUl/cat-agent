#!/bin/bash

set -e

cd /opt/llama.cpp

exec ./build-vulkan/bin/llama-server \
  -m /opt/llama.cpp/models/gemma-4/gemma-4-E4B-it-Q4_K_M.gguf \
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
  --spec-type draft-mtp \
  --spec-draft-model /opt/llama.cpp/models/gemma-4/mtp-gemma-4-E4B-it.gguf \
  --spec-draft-device none \
  --spec-draft-ngl 0 \
  --spec-draft-n-max 3

