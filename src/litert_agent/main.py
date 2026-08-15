from __future__ import annotations

import logging
from pathlib import Path
import sys

import litert_lm

from cat_agent.config import Settings

from .runtime import build_bundle, warm_bundle
from .tui import LiteRTTUI

LOGGER = logging.getLogger(__name__)
LOG_DIR = Path("/var/log/litertlm")
LOG_PATH = LOG_DIR / "cat-agent.log"


def main() -> int:
    settings = Settings.from_env(require_model=False)
    try:
        console_handler = _configure_logging(settings)
    except OSError as exc:
        print(f"Cannot initialize log {LOG_PATH}: {exc}", file=sys.stderr)
        return 2

    # Keep LiteRT-LM's validated native startup/profiler warnings out of the
    # human-facing console. Python-side diagnostics are retained in cat-agent.log.
    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    LOGGER.info("cat-agent LiteRT backend starting")
    LOGGER.info("Log file: %s", LOG_PATH)
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

        _remove_console_handler(console_handler)
        LiteRTTUI(bundle).run()
        return 0
    finally:
        bundle.close()
        LOGGER.info("cat-agent LiteRT backend stopped")


def _configure_logging(settings: Settings) -> logging.Handler:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(threadName)s]: %(message)s"
    )
    level = getattr(logging, settings.log_level, logging.INFO)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    return console_handler


def _remove_console_handler(handler: logging.Handler) -> None:
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
