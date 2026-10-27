"""Independent native-tools CORE IPC: optional serialized Conversation manager.

No Task SYSTEM, legacy scheduler, or agent imports. Tool calls are native.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
from orchestration_tools.contracts import ManagerRuntime


LOGGER = logging.getLogger(__name__)
DEFAULT_SOCKET = Path("/run/cat-agent/litert-tools.sock")
MAX_FRAME_BYTES = 65536
MAX_CLIENTS = 16
MAX_PENDING = 16


@dataclass(eq=False)
class _Client:
    sock: socket.socket
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "unknown"
    epoch: int = 0
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
    """Dedicated socket with one owner and one serialized model worker."""

    def __init__(self, path: Path = DEFAULT_SOCKET, *, manager: ManagerRuntime | None = None) -> None:
        self.path = Path(path)
        self._socket_owner = SocketOwner(self.path)
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._clients: dict[_Client, threading.Thread] = {}
        self._human: _Client | None = None
        self._manager = manager
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="native-manager") if manager else None
        self._pending = 0
        self._working = False

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
        if self._manager is None:
            LOGGER.info("TOOLS CORE SKELETON READY: %s (manager not connected)", self.path)
        else:
            LOGGER.info("TOOLS CORE READY: %s (native Conversation manager)", self.path)

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
                released = self._human is client
                if released:
                    self._human = None
                    client.epoch += 1
                    self._schedule_release(client.session_id)
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
        with self._lock:
            if self._manager is None:
                return {
                    "state": "NOT_READY", "ready": False, "pending": 0,
                    "label": "", "reason": "manager_not_connected_stage1",
                }
            status = {
                "state": "BUSY" if self._working else "IDLE", "ready": True,
                "pending": self._pending, "label": "manager" if self._working else "",
            }
            if hasattr(self._manager, "agent_status"):
                status["agents"] = self._manager.agent_status()
            return status

    def _schedule_release(self, session_id: str) -> None:
        # Called while holding _lock; serialized AFTER in-flight inference.
        if self._executor is not None:
            self._executor.submit(self._manager.release_session, session_id)

    def _model_worker(
        self, client: _Client, epoch: int, request_id: str, text: str,
        agent_id: str | None = None,
    ) -> None:
        with self._lock:
            self._pending -= 1
            if self._human is not client or client.epoch != epoch or self._stop.is_set():
                return
            self._working = True
        try:
            try:
                if agent_id is None:
                    turn = self._manager.respond(client.session_id, text)
                else:
                    from orchestration_tools.agents import AgentState
                    from orchestration_tools.contracts import ManagerTurn
                    outcome = self._manager.run_agent(agent_id, text)
                    turn = ManagerTurn(
                        "reply" if outcome.state is AgentState.FREE else "error",
                        outcome.text,
                    )
            except Exception as exc:
                LOGGER.exception("Native manager turn failed id=%s", request_id)
                # This response may reflect a partially mutated native KV. Reset
                # serialized after the exception, never replay a tool call.
                if agent_id is None:
                    try:
                        self._manager.release_session(client.session_id)
                    except Exception:
                        LOGGER.exception("Failed to reset native manager after error")
                from orchestration_tools.contracts import ManagerTurn
                turn = ManagerTurn("error", str(exc))
            with self._lock:
                if self._human is not client or client.epoch != epoch or self._stop.is_set():
                    return
                fields = {
                    "request_id": request_id, "session_id": client.session_id,
                    "kind": turn.kind, "text": turn.text, "chat_open": agent_id is None,
                }
                if agent_id is not None:
                    fields["agent_id"] = agent_id
                if turn.kind == "error":
                    client.send({"type": "error", **fields})
                elif turn.kind == "reply":
                    client.send({"type": "reply", **fields})
                client.send({"type": "completed", **fields})
        finally:
            with self._lock:
                self._working = False

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
                    if self._human is not client:
                        client.epoch += 1
                    self._human = client
                    client.name = name
                owner = self._human.name if self._human else None
            if acquired:
                client.send({
                    "type": "acquired", "client": name,
                    "session_id": client.session_id,
                    "core": {
                        "socket": str(self.path),
                        "model": getattr(self._manager, "model_name", None),
                        "backend": "UNCONNECTED" if self._manager is None else "CPU",
                        "stage": 1 if self._manager is None else 2,
                        "manager_warm_tokens": getattr(self._manager, "base_kv_tokens", None),
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
                    client.epoch += 1
                    self._schedule_release(client.session_id)
            client.send({"type": "released"})
            return
        if message_type in ("user", "agent_task"):
            content = message.get("text")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("user text must be nonempty")
            # json.loads permits escaped lone surrogates; C++ LiteRT does not.
            content.encode("utf-8", errors="strict")
            agent_id = None
            if message_type == "agent_task":
                agent_id = message.get("agent")
                if not isinstance(agent_id, str) or not agent_id.strip():
                    raise ValueError("agent_task requires a non-empty agent id")
                if self._manager is not None:
                    pool_status = getattr(self._manager, "agent_status", lambda: [])()
                    if not any(a["agent_id"] == agent_id for a in pool_status):
                        raise ValueError(f"agent is not configured: {agent_id}")
            with self._lock:
                owner = self._human is client
                current = self._human.name if self._human else None
            if not owner:
                client.send({"type": "busy", "owner": current, "text": "Гена занят"})
            elif self._manager is None:
                client.send({
                    "type": "error", "error": "manager backend is not connected (stage 1)",
                })
            else:
                with self._lock:
                    if self._human is not client:
                        client.send({"type": "busy", "owner": None, "text": "Гена занят"})
                        return
                    if self._pending >= MAX_PENDING:
                        client.send({
                            "type": "error", "error": "CORE request queue is full",
                        })
                        return
                    request_id = uuid.uuid4().hex
                    client.send({
                        "type": "accepted", "request_id": request_id,
                        "session_id": client.session_id,
                    })
                    self._pending += 1
                    self._executor.submit(
                        self._model_worker, client, client.epoch,
                        request_id, content, agent_id,
                    )
            return
        # Voice, events, TASK SYSTEM and fallback channels are not connected.
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
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=False)
                self._executor = None
            if self._manager is not None:
                self._manager.close()
            self._socket_owner.release()
        LOGGER.info("TOOLS CORE stopped: %s", self.path)
