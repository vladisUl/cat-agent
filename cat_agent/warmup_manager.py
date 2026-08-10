from __future__ import annotations

import logging
import sys

from .config import Settings
from .main import build_runtime

LOGGER = logging.getLogger(__name__)


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    response = runtime.client.chat(runtime.messages)
    cached = response.cached_tokens
    new = response.prompt_evaluated_tokens
    prefill = response.prompt_seconds

    print(
        "manager warmup: "
        f"{response.elapsed_seconds:.3f}s, "
        f"cached={cached if cached is not None else '?'}, "
        f"new={new if new is not None else '?'}, "
        f"prefill={prefill:.3f}s" if prefill is not None else
        f"manager warmup: {response.elapsed_seconds:.3f}s, "
        f"cached={cached if cached is not None else '?'}, "
        f"new={new if new is not None else '?'}, prefill=?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
