# Sourced by interface launchers before changing directory.
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 litert|openai" >&2
    exit 2
fi
case "$1" in
    litert|openai) ;;
    *) echo "Unknown CORE: $1. Expected litert or openai." >&2; exit 2 ;;
esac
export CAT_AGENT_CORE_SOCKET="${CAT_AGENT_CORE_SOCKET:-/run/cat-agent/$1.sock}"
