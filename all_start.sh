#!/usr/bin/env bash
set -euo pipefail

systemctl start \
    cat-agent-task-system.service \
    cat-agent-litert.service \
    cat-agent-openai.service
