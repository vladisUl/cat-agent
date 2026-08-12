from __future__ import annotations

import logging
import sys

from cat_agent.config import Settings

from .runtime import build_bundle, warm_bundle

LOGGER = logging.getLogger(__name__)


def main() -> int:
    settings = Settings.from_env(require_model=False)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    LOGGER.info("cat-agent LiteRT backend starting")
    LOGGER.info("Workspace: %s", settings.workspace)
    LOGGER.info("Prompt dir: %s", settings.prompt_dir)

    bundle = build_bundle(settings)
    try:
        manager_warm, agent_warm = warm_bundle(bundle, settings)
        LOGGER.info(
            "LiteRT prefixes ready: manager=%d tokens %.3fs, agent=%d tokens %.3fs",
            manager_warm.token_count,
            manager_warm.elapsed_seconds,
            agent_warm.token_count,
            agent_warm.elapsed_seconds,
        )

        print("cat-agent LiteRT ready. Commands: /quit")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break

            turn = bundle.runtime.user_message(text)
            prefix = "manager" if turn.kind in {"reply", "ask"} else turn.kind
            print(f"{prefix}> {turn.text}")

        return 0
    finally:
        bundle.close()
        LOGGER.info("cat-agent LiteRT backend stopped")


if __name__ == "__main__":
    sys.exit(main())
