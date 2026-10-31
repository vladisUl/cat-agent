#!/usr/bin/env bash
# One shared TASK SYSTEM, two independently started COREs (OpenAI + native LiteRT).
# The sole Native CORE launcher; low-level CLI remains available via python -m agent_core_tools.
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

# Honor the active CPU/GPU model profile. Unsupported backends fail explicitly.
BACKEND="${LITERT_AGENT_BACKEND:-cpu}"
case "$BACKEND" in
    cpu|gpu) ;;
    *) echo "Unsupported Native LiteRT backend: $BACKEND (expected cpu or gpu)" >&2; exit 2 ;;
esac
# YNNPACK remains disabled in native LiteRT (0.18.x CPU UNIMPLEMENTED).

# The selected YAML profile owns Vulkan driver selection. Never silently
# inherit a previously exported WSL/Dozen adapter when switching profiles.
unset VK_DRIVER_FILES MESA_D3D12_DEFAULT_ADAPTER_NAME
if [[ -n "${LITERT_AGENT_VULKAN_ICD_DIR:-}" ]]; then
    ICD_DIR="$LITERT_AGENT_VULKAN_ICD_DIR"
    if [[ ! -d "$ICD_DIR" ]]; then
        echo "Native CORE Vulkan ICD directory missing: $ICD_DIR" >&2
        exit 2
    fi
    shopt -s nullglob
    icd_files=("$ICD_DIR"/*.json)
    shopt -u nullglob
    if (( ${#icd_files[@]} != 1 )); then
        echo "Native CORE expected exactly one Vulkan ICD JSON in $ICD_DIR; found ${#icd_files[@]}" >&2
        exit 2
    fi
    export VK_DRIVER_FILES="${icd_files[0]}"
    unset icd_files
fi
if [[ -n "${LITERT_AGENT_D3D12_ADAPTER:-}" ]]; then
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="$LITERT_AGENT_D3D12_ADAPTER"
fi
if [[ "$BACKEND" == "gpu" && -n "${VK_DRIVER_FILES:-}" ]]; then
    echo "Native CORE Vulkan ICD: $VK_DRIVER_FILES; D3D12 adapter: ${MESA_D3D12_DEFAULT_ADAPTER_NAME:-default}" >&2
fi

MODEL="${CAT_AGENT_NATIVE_MODEL_PATH:-${LITERT_AGENT_MODEL_PATH:-}}"
PROMPT="${CAT_AGENT_NATIVE_MANAGER_PROMPT:-${CAT_AGENT_PROMPTS_TOOLS_DIR:?Missing prompts_tools dir from YAML}/manager.md}"
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
    --backend "$BACKEND"
    --vision-backend "${LITERT_AGENT_VISION_BACKEND:-cpu}"
    --system-file "$PROMPT"
    --agent-config "$AGENTS"
    --command-dir "${CAT_AGENT_NATIVE_COMMAND_DIR:-/opt/model/bin}"
    --shell-cwd "$WORKSPACE"
    --enable-task-worker
    --enable-shell
)
if [[ "$BACKEND" == "cpu" && -n "${LITERT_AGENT_CPU_THREADS:-}" ]]; then
    args+=(--cpu-threads "$LITERT_AGENT_CPU_THREADS")
fi
if [[ -n "${LITERT_AGENT_ACTIVATION_DATA_TYPE:-}" ]]; then
    args+=(--activation-dtype "$LITERT_AGENT_ACTIVATION_DATA_TYPE")
fi
if [[ "${LITERT_AGENT_SPECULATIVE:-0}" == "1" ]]; then
    args+=(--speculative)
fi
if [[ "${CAT_AGENT_NATIVE_TRACE_JSON:-0}" == "1" ]]; then
    args+=(--trace-native-json)
fi

# No second TASK SYSTEM; native worker registers as executor=litert on the
# existing service. Old OpenAI CORE keeps executor=openai and its own socket.
NATIVE_PYTHON="${CAT_AGENT_TOOLS_PYTHON:-/opt/litert-lm-venv/bin/python3}"
exec "$NATIVE_PYTHON" -m agent_core_tools "${args[@]}"
