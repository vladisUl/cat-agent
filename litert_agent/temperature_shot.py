from __future__ import annotations

import logging
import sys
import time

from cat_agent.config import Settings

from .main import build_runtime

TASK = "Получить текущую температуру на улице."


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    started = time.monotonic()
    turn = runtime.user_message(TASK)
    total = time.monotonic() - started

    print(f"CONTROL_TASK={TASK}")
    print(f"CONTROL_RESULT_KIND={turn.kind}")
    print(f"CONTROL_RESULT={turn.text}")
    print(f"CONTROL_TOTAL={total:.3f}s")
    return 0 if turn.kind in {"reply", "ask"} else 2


if __name__ == "__main__":
    sys.exit(main())
