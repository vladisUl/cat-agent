#!/usr/bin/env bash
set -euo pipefail

systemctl stop \
    cat-agent-openai.service \
    cat-agent-litert.service \
    cat-agent-task-system.service
