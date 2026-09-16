from __future__ import annotations

import json
import os

from . import voice


def _load_wake_words() -> set[str]:
    raw = os.environ.get("CAT_AGENT_WAKE_WORDS_JSON", "")
    if not raw:
        return voice.WAKE_WORDS
    value = json.loads(raw)
    if not isinstance(value, list) or not value:
        raise ValueError("CAT_AGENT_WAKE_WORDS_JSON must contain a non-empty JSON list")
    words = {str(item).strip().lower() for item in value if str(item).strip()}
    if not words:
        raise ValueError("CAT_AGENT_WAKE_WORDS_JSON contains no usable wake words")
    return words


def main() -> None:
    voice.WAKE_WORDS = _load_wake_words()
    voice.NO_COMMAND_TIMEOUT_SECONDS = float(
        os.environ.get(
            "CAT_AGENT_NO_COMMAND_TIMEOUT_SECONDS",
            str(voice.NO_COMMAND_TIMEOUT_SECONDS),
        )
    )
    voice.MAX_COMMAND_SECONDS = float(
        os.environ.get(
            "CAT_AGENT_MAX_COMMAND_SECONDS",
            str(voice.MAX_COMMAND_SECONDS),
        )
    )
    voice.main()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSTOP")
    except Exception as exc:
        print(f"FATAL: {type(exc).__name__}: {exc}")
        raise
