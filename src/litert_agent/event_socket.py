from __future__ import annotations

from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import socket
import threading


LOGGER = logging.getLogger(__name__)
DEFAULT_EVENT_SOCKET = Path("/run/cat-agent/events.sock")
EventCallback = Callable[[str, str], None]


class ExternalEventSocket:
    """Receive local external events over a Unix datagram socket."""

    def __init__(
        self,
        callback: EventCallback,
        path: Path | None = None,
    ) -> None:
        self.callback = callback
        env_path = os.getenv("CAT_AGENT_EVENT_SOCKET", "").strip()
        self.path = Path(env_path) if env_path else (path or DEFAULT_EVENT_SOCKET)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.bind(str(self.path))
            os.chmod(self.path, 0o660)
            sock.settimeout(0.25)
        except Exception:
            sock.close()
            self.path.unlink(missing_ok=True)
            raise

        self._stop.clear()
        self._socket = sock
        self._thread = threading.Thread(
            target=self._run,
            name="cat-agent-events",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info("External event socket ready: %s", self.path)

    def close(self) -> None:
        self._stop.set()
        sock = self._socket
        if sock is not None:
            sock.close()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None
        self._socket = None
        self.path.unlink(missing_ok=True)
        LOGGER.info("External event socket stopped: %s", self.path)

    def _run(self) -> None:
        sock = self._socket
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                payload = sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                LOGGER.exception("External event socket receive failed")
                return

            try:
                item = json.loads(payload.decode("utf-8"))
                if not isinstance(item, dict):
                    raise TypeError("payload must be a JSON object")
                source = str(item["source"]).strip().lower()
                name = str(item["name"]).strip()
                if not source or not name:
                    raise ValueError("source and name must be non-empty")
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                LOGGER.warning("Ignored invalid external event payload: %s", exc)
                continue

            LOGGER.info("External event received source=%s name=%s", source, name)
            try:
                self.callback(source, name)
            except Exception:
                LOGGER.exception(
                    "External event callback failed source=%s name=%s",
                    source,
                    name,
                )
