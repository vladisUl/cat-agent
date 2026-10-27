"""Stage-1 Unix IPC CORE skeleton; deliberately does not run a model or TASKs.

Only basic legacy-compatible control messages are implemented. An unmatched
`user` request is never acknowledged as work; this server is NOT production
ready until the manager and scheduler are connected in later stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import socket
import threading
import uuid
from typing import Any

from .socket_owner import SocketOwner


LOGGER = logging.getLogger(__name__)
DEFAULT_SOCKET = Path("/run/cat-agent/litert-tools.sock")
MAX_FRAME_BYTES = 65536
MAX_CLIENTS = 16


@dataclass(eq=False)
class _Client:
    sock: socket.socket
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "unknown"
    send_lock: threading.Lock = field(default_factory=threading.Lock)

    def send(self, payload: dict[str, object]) -> None:
        data = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        with self.send_lock:
            self.sock.sendall(data)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


class CoreServer:
    """Dedicated socket, exclusive human owner, observable NOT_READY state."""

    def __init__(self, path: Path = DEFAULT_SOCKET) -> None:
        self.path = Path(path)
        self._socket_owner = SocketOwner(self.path)
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._clients: dict[_Client, threading.Thread] = {}
        self._human: _Client | None = None

    def start(self) -> None:
        with self._close_lock:
            if self._socket is not None:
                return
            self._socket_owner.acquire()
            listener = None
            try:
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                listener.bind(str(self.path))
                self._socket_owner.bound()
                os.chmod(self.path, 0o660)
                listener.listen(MAX_CLIENTS)
                listener.settimeout(0.25)
            except BaseException:
                if listener is not None:
                    listener.close()
                self._socket_owner.release()
                raise
            self._stop.clear()
            self._socket = listener
        LOGGER.info("TOOLS CORE SKELETON READY: %s (manager not connected)", self.path)

    def serve_forever(self) -> None:
        if self._socket is None:
            self.start()
        while not self._stop.is_set():
            listener = self._socket
            if listener is None:
                break
            try:
                sock, _ = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            sock.settimeout(0.5)
            client = _Client(sock)
            with self._lock:
                if len(self._clients) >= MAX_CLIENTS:
                    client.close()
                    continue
                worker = threading.Thread(
                    target=self._client_loop, args=(client,),
                    name="cat-agent-tools-ipc", daemon=True,
                )
                self._clients[client] = worker
            worker.start()

    def _client_loop(self, client: _Client) -> None:
        data = bytearray()
        try:
            while not self._stop.is_set():
                try:
                    chunk = client.sock.recv(8192)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                data.extend(chunk)
                while b"\n" in data:
                    raw, _, rest = data.partition(b"\n")
                    data = bytearray(rest)
                    if len(raw) > MAX_FRAME_BYTES:
                        client.send({"type": "error", "error": "IPC message exceeds 64 KiB"})
                        return
                    if raw.strip():
                        self._process_frame(client, raw)
                if len(data) > MAX_FRAME_BYTES:
                    client.send({"type": "error", "error": "IPC message exceeds 64 KiB"})
                    return
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            with self._lock:
                if self._human is client:
                    self._human = None
                self._clients.pop(client, None)
            client.close()

    def _process_frame(self, client: _Client, raw: bytes) -> None:
        try:
            message = json.loads(raw.decode("utf-8", errors="strict"))
            if not isinstance(message, dict):
                raise ValueError("message must be a JSON object")
            self._handle_message(client, message)
        except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
            client.send({"type": "error", "error": f"invalid message: {exc}"})

    def _status(self) -> dict[str, object]:
        return {
            "state": "NOT_READY", "ready": False, "pending": 0,
            "label": "", "reason": "manager_not_connected_stage1",
        }

    def _handle_message(self, client: _Client, message: dict[str, Any]) -> None:
        message_type = message.get("type")
        if not isinstance(message_type, str) or not message_type.strip():
            raise ValueError("type must be a nonempty string")
        message_type = message_type.strip().lower()
        if message_type == "ping":
            client.send({"type": "pong"})
            return
        if message_type == "snapshot":
            client.send({"type": "snapshot", "status": self._status()})
            return
        if message_type == "who":
            with self._lock:
                owner = self._human.name if self._human else None
            client.send({"type": "session", "owner": owner})
            return
        if message_type == "acquire":
            name = message.get("client", "unknown")
            if not isinstance(name, str):
                raise ValueError("client must be a string")
            name = name.strip() or "unknown"
            with self._lock:
                acquired = self._human is None or self._human is client
                if acquired:
                    self._human = client
                    client.name = name
                owner = self._human.name if self._human else None
            if acquired:
                client.send({
                    "type": "acquired", "client": name,
                    "session_id": client.session_id,
                    "core": {
                        "socket": str(self.path),
                        "model": None,
                        "backend": "UNCONNECTED",
                        "stage": 1,
                    },
                    "status": self._status(),
                })
            else:
                client.send({"type": "busy", "owner": owner, "text": "Гена занят"})
            return
        if message_type == "release":
            with self._lock:
                if self._human is client:
                    self._human = None
            client.send({"type": "released"})
            return
        if message_type == "user":
            content = message.get("text")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("user text must be nonempty")
            # json.loads permits escaped lone surrogates; C++ LiteRT does not.
            content.encode("utf-8", errors="strict")
            with self._lock:
                owner = self._human is client
                current = self._human.name if self._human else None
            if not owner:
                client.send({"type": "busy", "owner": current, "text": "Гена занят"})
            else:
                client.send({
                    "type": "error", "error": "manager backend is not connected (stage 1)",
                })
            return
        # Stage 1 has no voice, event, fallback, MODEL or TASK execution.
        client.send({
            "type": "error", "error": f"{message_type} is not available in stage 1",
        })

    def close(self) -> None:
        with self._close_lock:
            self._stop.set()
            listener, self._socket = self._socket, None
            if listener is not None:
                listener.close()
            with self._lock:
                clients = tuple(self._clients)
                threads = tuple(self._clients.values())
                self._human = None
            for client in clients:
                client.close()
            for thread in threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=1.0)
            self._socket_owner.release()
        LOGGER.info("TOOLS CORE stopped: %s", self.path)
