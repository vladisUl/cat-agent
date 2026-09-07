#!/usr/bin/env bash
set -euo pipefail

cd /opt/cat-agent
export CAT_AGENT_WEB_HOST=0.0.0.0
export CAT_AGENT_WEB_TRUSTED_NETWORK=1
export CAT_AGENT_WEB_ORIGINS='https://vlad777.ddns.net,http://192.168.0.140:8080'
exec ./start_web.sh
