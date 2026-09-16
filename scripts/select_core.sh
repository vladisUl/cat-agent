# Sourced by interface launchers before changing directory.
# No argument: select the preferred READY CORE (openai, then litert).
# One argument: explicit administrative override.
# CAT_AGENT_CORE_SOCKET remains the lowest-level socket override.

_repo_root="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [litert|openai]" >&2
    return 2 2>/dev/null || exit 2
fi

if [[ $# -eq 1 ]]; then
    case "$1" in
        litert|openai) _selected_core="$1" ;;
        *)
            echo "Unknown CORE: $1. Expected litert or openai." >&2
            return 2 2>/dev/null || exit 2
            ;;
    esac
else
    _selected_core=""
fi

# Preserve the existing low-level debugging/administrative override, but only
# after validating launcher arguments.
if [[ -n "${CAT_AGENT_CORE_SOCKET:-}" ]]; then
    unset _repo_root _selected_core
    return 0 2>/dev/null || exit 0
fi

if [[ -z "$_selected_core" ]]; then
    if ! _core_status="$("$_repo_root/task_system.sh" cores 2>/dev/null)"; then
        echo "Cannot query Task SYSTEM CORE readiness." >&2
        return 1 2>/dev/null || exit 1
    fi

    if grep -Eq '^openai[[:space:]]+READY([[:space:]]|$)' <<<"$_core_status"; then
        _selected_core="openai"
    elif grep -Eq '^litert[[:space:]]+READY([[:space:]]|$)' <<<"$_core_status"; then
        _selected_core="litert"
    else
        echo "No READY CORE available." >&2
        printf '%s\n' "$_core_status" >&2
        return 1 2>/dev/null || exit 1
    fi
    echo "Selected CORE: $_selected_core (auto)" >&2
fi

export CAT_AGENT_CORE_SOCKET="/run/cat-agent/${_selected_core}.sock"
unset _repo_root _selected_core _core_status
