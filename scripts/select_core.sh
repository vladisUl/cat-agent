# Sourced by interface launchers before changing directory.
# No argument: select the preferred READY CORE (openai, then litert).
# One argument: explicit administrative override.

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
