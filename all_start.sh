#!/usr/bin/env bash
set -euo pipefail

systemctl start \
    cat-agent-system-core.service \
    cat-agent-litert.service \
    cat-agent-openai.service
