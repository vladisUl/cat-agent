from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
import select
import socket
import threading
import time
import uuid

LOGGER = logging.getLogger(__name__)


@dataclass(eq=False)
class ClientConnection:
    sock: socket.socket
    address: object = None
    client_name: str = "unknown"
    human_owner: bool = False
    fallback: bool = False
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    capacity: int = 256
    send_timeout: float = 3.0
    _queue: deque = field(default_factory=deque)
    _condition: threading.Condition = field(default_factory=threading.Condition)
    _closed: bool = False
    _thread: object = None

    def start(self):
        self._thread = threading.Thread(target=self._write_loop, name="cat-agent-send", daemon=True)
        self._thread.start()

    def send(self, payload):
        with self._condition:
            if self._closed:
                return False
            if payload.get("type") == "status":
                self._queue = deque(p for p in self._queue if p.get("type") != "status")
            if len(self._queue) >= self.capacity:
                LOGGER.warning("CORE slow client disconnected session=%s", self.session_id)
                self.close()
                return False
            self._queue.append(payload)
            self._condition.notify()
            return True

    def _write_loop(self):
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._closed or self._queue)
                    if self._closed:
                        return
                    payload = self._queue.popleft()
                data = memoryview((json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                deadline = time.monotonic() + self.send_timeout
                while data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("client send timed out")
                    if not select.select([], [self.sock], [], min(remaining, 0.2))[1]:
                        continue
                    try:
                        sent = self.sock.send(data, socket.MSG_DONTWAIT)
                    except BlockingIOError:
                        continue
                    if not sent:
                        raise ConnectionError("client closed")
                    data = data[sent:]
        except (OSError, ValueError):
            LOGGER.info("CORE writer disconnected session=%s", self.session_id)
        finally:
            self.close()

    def close(self):
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._queue.clear()
            self._condition.notify_all()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
