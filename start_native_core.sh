#!/usr/bin/env bash
# One shared TASK SYSTEM, two independently started COREs (OpenAI + native LiteRT).
# Explicit arguments/testing remain available through start_litert_agent_tools.sh.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# The existing configuration parser is installed in the legacy venv;
# native LiteRT venv need not have PyYAML installed.
CONFIG_PYTHON="${CAT_AGENT_NATIVE_CONFIG_PYTHON:-/opt/litert-lm-venv/bin/python3}"
if ! CONFIG_ENV="$("$CONFIG_PYTHON" -m orchestration.config_env litert)"; then
    echo "Cannot load active LiteRT profile from cat-agent.yaml" >&2
    exit 2
fi
eval "$CONFIG_ENV"
unset CONFIG_ENV

# Native Engine is CPU-only. Do not silently pretend to support GPU,
# YNNPACK or speculative settings from the legacy LiteRT profile.
if [[ "${LITERT_AGENT_BACKEND:-}" != "cpu" ]]; then
    echo "Native LiteRT CORE currently requires a CPU active profile (litert.active_profile)." >&2
    exit 2
fi

MODEL="${CAT_AGENT_NATIVE_MODEL_PATH:-${LITERT_AGENT_MODEL_PATH:-}}"
PROMPT="${CAT_AGENT_NATIVE_MANAGER_PROMPT:-${ROOT}/prompts_tools/manager.md}"
AGENTS="${CAT_AGENT_NATIVE_AGENT_CONFIG:-${ROOT}/config/native_agents.example.json}"
WORKSPACE="${CAT_AGENT_NATIVE_SHELL_CWD:-${CAT_AGENT_WORKSPACE:-/opt/model}}"
CORE_SOCKET="${CAT_AGENT_NATIVE_CORE_SOCKET:-/run/cat-agent/litert-tools.sock}"
TASK_SOCKET="${CAT_AGENT_TASK_SOCKET:-/run/cat-agent/tasks.sock}"

for entry in "model:$MODEL" "prompt:$PROMPT" "agents:$AGENTS"; do
    label="${entry%%:*}"
    path="${entry#*:}"
    if [[ ! -f "$path" ]]; then
        echo "Native CORE $label file missing: $path" >&2
        echo "Use CAT_AGENT_NATIVE_MODEL_PATH or update litert.active_profile in cat-agent.yaml." >&2
        exit 2
    fi
done
if [[ "$CORE_SOCKET" != /* || "$TASK_SOCKET" != /* ]]; then
    echo "Native CORE and TASK SYSTEM sockets must be absolute paths" >&2
    exit 2
fi
if [[ "$CORE_SOCKET" == "$TASK_SOCKET" ||
      "$CORE_SOCKET" == "/run/cat-agent/openai.sock" ||
      "$CORE_SOCKET" == "/run/cat-agent/litert.sock" ]]; then
    echo "Native CORE socket must be distinct from TASK SYSTEM and legacy CORE sockets" >&2
    exit 2
fi

args=(
    --socket "$CORE_SOCKET"
    --task-socket "$TASK_SOCKET"
    --model "$MODEL"
    --system-file "$PROMPT"
    --agent-config "$AGENTS"
    --shell-cwd "$WORKSPACE"
    --enable-task-worker
    --enable-shell
    --disable-mqtt
)
if [[ -n "${LITERT_AGENT_CPU_THREADS:-}" ]]; then
    args+=(--cpu-threads "$LITERT_AGENT_CPU_THREADS")
fi

# No second TASK SYSTEM; native worker registers as executor=litert on the
# existing service. Old OpenAI CORE keeps executor=openai and its own socket.
exec "${ROOT}/start_litert_agent_tools.sh" "${args[@]}"
