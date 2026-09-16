from __future__ import annotations

import logging
from pathlib import Path
import sys

from agent_core.core_server import CoreServer
from agent_core.systemd_notify import notify_systemd
from llama_agent.main import _ProtocolLogFilter
from orchestration.config import Settings

from .runtime import build_bundle

LOGGER = logging.getLogger(__name__)
LOG_DIR = Path("/var/log/litertlm")
LOG_PATH = LOG_DIR / "cat-agent.log"


def main() -> int:
    settings = Settings.from_env()
    try:
        _configure_logging(settings)
    except OSError as exc:
        print(f"Cannot initialize log {LOG_PATH}: {exc}", file=sys.stderr)
        return 2

    LOGGER.info("cat-agent CORE starting backend=OpenAI-compatible")
    LOGGER.info("Log file: %s", LOG_PATH)
    LOGGER.info("Workspace: %s", settings.workspace)
    LOGGER.info("Prompt dir: %s", settings.prompt_dir)
    LOGGER.info("Model endpoint: %s model=%s", settings.api_base_url, settings.model)

    notify_systemd(status="Initializing OpenAI CORE runtime")
    bundle = build_bundle(settings)
    try:
        notify_systemd(status="Starting OpenAI CORE socket")
        core = CoreServer(bundle, path=Path("/run/cat-agent/openai.sock"))
        core.start()
        notify_systemd(status="OpenAI CORE ready", ready=True)
        try:
            core.serve_forever()
        finally:
            notify_systemd(status="Stopping OpenAI CORE", stopping=True)
            core.close()
        return 0
    finally:
        bundle.close()
        LOGGER.info("cat-agent CORE stopped backend=OpenAI-compatible")


def _configure_logging(settings: Settings) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    level = getattr(logging, settings.log_level, logging.INFO)
    protocol_filter = _ProtocolLogFilter()

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(protocol_filter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(protocol_filter)
    root.addHandler(console_handler)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
