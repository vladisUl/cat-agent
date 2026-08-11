from __future__ import annotations

import logging
import sys
import time

from cat_agent.config import Settings

from .native_main import build_native_runtime, warm_native_runtime

TASK = "Получить текущую температуру на улице."


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bundle = build_native_runtime(settings)
    try:
        warmup = warm_native_runtime(bundle, settings, agent_skills=("mqtt",))

        started = time.monotonic()
        turn = bundle.runtime.user_message(TASK)
        total = time.monotonic() - started

        print(f"CONTROL_TASK={TASK}")
        print(f"CONTROL_RESULT_KIND={turn.kind}")
        print(f"CONTROL_RESULT={turn.text}")
        print(f"CONTROL_ENGINE_INIT={bundle.engine_init_seconds:.3f}s")
        print(f"CONTROL_WARMUP_MANAGER={warmup.manager_seconds:.3f}s")
        print(f"CONTROL_WARMUP_AGENT={warmup.agent_seconds:.3f}s")
        print(f"CONTROL_WARMUP_TOTAL={warmup.total_seconds:.3f}s")
        print(f"CONTROL_CHAIN_TOTAL={total:.3f}s")
        return 0 if turn.kind in {"reply", "ask"} else 2
    finally:
        bundle.close()


if __name__ == "__main__":
    sys.exit(main())
