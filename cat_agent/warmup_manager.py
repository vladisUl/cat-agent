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
    print(
        "manager warmup: "
        f"{response.elapsed_seconds:.3f}s, "
        f"prompt_tokens={response.prompt_tokens if response.prompt_tokens is not None else '?'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
