"""Standalone LiteRT CORE logging: technical events and native conversation transcript.

The OpenAI CORE uses its own logging setup and file, untouched here.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path


DEFAULT_LOG_PATH = Path("/var/log/litertlm/cat-agent-litert.log")


def configure_native_logging(level: str, *, path: Path | None = None) -> Path:
    """Attach file and stderr handlers for the standalone LiteRT process.

    The optional environment override is for isolated tests / deployments.
    The configured log contains user text and tool results: protect its access.
    """
    destination = (
        Path(path) if path is not None
        else Path(os.getenv("CAT_AGENT_LITERT_LOG_PATH", str(DEFAULT_LOG_PATH)))
    )
    if not destination.is_absolute():
        raise ValueError("CAT_AGENT_LITERT_LOG_PATH must be an absolute path")
    level_name = level.upper()
    if level_name not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"Invalid native log level: {level!r}")
    log_level = getattr(logging, level_name)

    destination.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(destination, encoding="utf-8")
    try:
        # Operator policy: readable by all local users (contains user/tool transcript).
        os.chmod(destination, 0o644)
    except OSError:
        file_handler.close()
        raise
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(log_level)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    return destination
