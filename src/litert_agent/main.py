from __future__ import annotations

import logging
from pathlib import Path
import sys

import litert_lm

from cat_agent.config import Settings

from .runtime import build_bundle, warm_bundle
from .tui import LiteRTTUI

LOGGER = logging.getLogger(__name__)


def main() -> int:
    settings = Settings.from_env(require_model=False)
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Keep LiteRT-LM's validated native startup/profiler warnings out of the
    # human-facing console. Python-side diagnostics are retained in the log.
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

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

        _switch_logging_to_file(settings)
        LiteRTTUI(bundle).run()
        return 0
    finally:
        bundle.close()
        LOGGER.info("cat-agent LiteRT backend stopped")


def _switch_logging_to_file(settings: Settings) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    project_root = Path(__file__).resolve().parents[2]
    log_path = project_root / "cat-agent-litert.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))


if __name__ == "__main__":
    sys.exit(main())
