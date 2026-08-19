from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import socket
import threading
from typing import Any

from orchestration.manager import ManagerTurn

from .core_scheduler import CoreScheduler, HARDWARE_EVENT_PRIORITY
from .model_client import InferenceTiming

LOGGER = logging.getLogger(__name__)
DEFAULT_CORE_SOCKET = Path("/run/cat-agent/core.sock")


@dataclass(slots=True)
class _ClientConnection:
    sock: socket.socket
    address: object
    client_name: str = "unknown"
    human_owner: bool = False
    fallback: bool = False
    send_lock: threading.Lock = field(default_factory=threading.Lock)

    def send(self, payload: dict[str, object]) -> None:
        data = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self.send_lock:
            self.sock.sendall(data)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class CoreServer:
    """Local IPC boundary around the long-lived model/runtime core."""

    def __init__(
        self,
        bundle,
        *,
        path: Path | None = None,
        scheduler: CoreScheduler | None = None,
    ) -> None:
        env_path = os.getenv("CAT_AGENT_CORE_SOCKET", "").strip()
        self.path = Path(env_path) if env_path else (path or DEFAULT_CORE_SOCKET)
        self.bundle = bundle
        self.scheduler = scheduler or CoreScheduler(
            bundle,
            on_human_turn=self._deliver_human_turn,
            on_notification=self._deliver_notification,
            on_status=self._deliver_status,
            on_model_event=self._deliver_model_event,
        )

        self._lock = threading.RLock()
        self._human: _ClientConnection | None = None
        self._fallback: _ClientConnection | None = None
        self._clients: list[_ClientConnection] = []
        self._server_socket: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._server_socket is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(str(self.path))
            os.chmod(self.path, 0o660)
            sock.listen(16)
            sock.settimeout(0.25)
        except Exception:
            sock.close()
            self.path.unlink(missing_ok=True)
            raise

        self._stop.clear()
        self._server_socket = sock
        self.scheduler.start()
        LOGGER.info("CORE socket ready: %s", self.path)

    def serve_forever(self) -> None:
        if self._server_socket is None:
            self.start()
        sock = self._server_socket
        assert sock is not None

        while not self._stop.is_set():
            try:
                client_sock, address = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise

            connection = _ClientConnection(client_sock, address)
            with self._lock:
                self._clients.append(connection)
            thread = threading.Thread(
                target=self._client_loop,
                args=(connection,),
                name="cat-agent-ipc",
                daemon=True,
            )
            thread.start()

    def close(self) -> None:
        self._stop.set()
        sock = self._server_socket
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        self._server_socket = None

        with self._lock:
            clients = list(self._clients)
            self._human = None
            self._fallback = None
        for client in clients:
            client.close()

        self.scheduler.close()
        self.path.unlink(missing_ok=True)
        LOGGER.info("CORE socket stopped: %s", self.path)

    def session_owner(self) -> str | None:
        with self._lock:
            return self._human.client_name if self._human is not None else None

    def _client_loop(self, client: _ClientConnection) -> None:
        LOGGER.info("CORE client connected")
        try:
            reader = client.sock.makefile("r", encoding="utf-8", newline="\n")
            try:
                for raw in reader:
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if not isinstance(item, dict):
                            raise TypeError("message must be a JSON object")
                        self._handle_message(client, item)
                    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                        self._safe_send(
                            client,
                            {"type": "error", "error": f"invalid message: {exc}"},
                        )
            finally:
                reader.close()
        except OSError:
            LOGGER.info("CORE client connection closed")
        finally:
            self._detach_client(client)
            client.close()

    def _handle_message(self, client: _ClientConnection, item: dict[str, Any]) -> None:
        message_type = str(item.get("type", "")).strip().lower()
        if not message_type:
            raise ValueError("type must be non-empty")

        if message_type == "acquire":
            self._acquire_human(client, str(item.get("client", "unknown")))
            return

        if message_type == "release":
            self._release_human(client)
            self._safe_send(client, {"type": "released"})
            return

        if message_type == "register_fallback":
            self._register_fallback(client, str(item.get("client", "telegram")))
            return

        if message_type == "user":
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError("user text must be non-empty")
            with self._lock:
                owner = self._human is client
                current = self._human.client_name if self._human is not None else None
            if not owner:
                self._safe_send(
                    client,
                    {"type": "busy", "owner": current, "text": "Гена занят"},
                )
                return
            self.scheduler.submit_user(text)
            return

        if message_type == "event":
            source = str(item.get("source", "")).strip().lower()
            name = str(item.get("name", "")).strip()
            if not source or not name:
                raise ValueError("event source and name must be non-empty")
            event = self.bundle.runtime.external_event(source, name)
            if event is None:
                self._safe_send(
                    client,
                    {
                        "type": "event_rejected",
                        "source": source,
                        "name": name,
                    },
                )
                return
            self.scheduler.enqueue_external_event(
                event,
                priority=HARDWARE_EVENT_PRIORITY,
                coalesce=False,
            )
            self._safe_send(
                client,
                {"type": "event_accepted", "source": source, "name": name},
            )
            return

        if message_type == "snapshot":
            self._safe_send(
                client,
                {"type": "snapshot", "status": self._telemetry_snapshot()},
            )
            return

        if message_type == "who":
            self._safe_send(
                client,
                {"type": "session", "owner": self.session_owner()},
            )
            return

        if message_type == "ping":
            self._safe_send(client, {"type": "pong"})
            return

        raise ValueError(f"unknown message type: {message_type}")

    def _acquire_human(self, client: _ClientConnection, name: str) -> None:
        client_name = name.strip() or "unknown"
        with self._lock:
            if self._human is None or self._human is client:
                self._human = client
                client.client_name = client_name
                client.human_owner = True
                owner = client_name
                acquired = True
            else:
                owner = self._human.client_name
                acquired = False

        if acquired:
            LOGGER.info("CORE human session acquired by %s", owner)
            self._safe_send(
                client,
                {
                    "type": "acquired",
                    "client": owner,
                    "core": self._core_info(),
                    "status": self._telemetry_snapshot(),
                },
            )
        else:
            LOGGER.info("CORE human session busy owner=%s requested_by=%s", owner, client_name)
            self._safe_send(
                client,
                {"type": "busy", "owner": owner, "text": "Гена занят"},
            )

    def _release_human(self, client: _ClientConnection) -> None:
        with self._lock:
            if self._human is client:
                LOGGER.info("CORE human session released by %s", client.client_name)
                self._human = None
            client.human_owner = False

    def _register_fallback(self, client: _ClientConnection, name: str) -> None:
        client_name = name.strip() or "telegram"
        with self._lock:
            previous = self._fallback
            self._fallback = client
            client.client_name = client_name
            client.fallback = True
            if previous is not None and previous is not client:
                previous.fallback = False
        LOGGER.info("CORE fallback notification channel=%s", client_name)
        self._safe_send(
            client,
            {"type": "fallback_registered", "client": client_name},
        )

    def _detach_client(self, client: _ClientConnection) -> None:
        with self._lock:
            if self._human is client:
                LOGGER.info("CORE human session auto-release client=%s", client.client_name)
                self._human = None
            if self._fallback is client:
                LOGGER.info("CORE fallback channel disconnected client=%s", client.client_name)
                self._fallback = None
            try:
                self._clients.remove(client)
            except ValueError:
                pass
            client.human_owner = False
            client.fallback = False

    def _deliver_human_turn(self, turn: ManagerTurn) -> None:
        with self._lock:
            target = self._human
        if target is None:
            LOGGER.warning("CORE human reply has no active human client: %r", turn.text)
            return
        self._safe_send(
            target,
            {"type": "reply", "kind": turn.kind, "text": turn.text},
        )

    def _deliver_notification(self, turn: ManagerTurn) -> None:
        with self._lock:
            target = self._human if self._human is not None else self._fallback
            route = (
                "human"
                if self._human is not None
                else ("fallback" if self._fallback is not None else "none")
            )
        if target is None:
            LOGGER.warning("CORE notification dropped: no human/fallback route text=%r", turn.text)
            return
        LOGGER.info("CORE notification route=%s target=%s", route, target.client_name)
        self._safe_send(
            target,
            {"type": "notification", "kind": turn.kind, "text": turn.text},
        )

    def _deliver_status(self, status: dict[str, object]) -> None:
        with self._lock:
            target = self._human
        if target is not None:
            self._safe_send(target, {"type": "status", **status})

    def _deliver_model_event(
        self,
        label: str,
        event: str,
        payload: str,
        timing: InferenceTiming,
    ) -> None:
        with self._lock:
            target = self._human
        if target is None:
            return
        self._safe_send(
            target,
            {
                "type": "model_event",
                "label": label,
                "event": event,
                "payload": payload,
                "timing": self._timing_dict(timing),
            },
        )

    def _core_info(self) -> dict[str, object]:
        manager_warm = self.bundle.manager_warm
        agent_warm = self.bundle.agent_warm
        return {
            "socket": str(self.path),
            "model": self.bundle.model_path.name,
            "backend": self.bundle.backend_name.upper(),
            "speculative": self.bundle.speculative,
            "manager_engine_seconds": self.bundle.manager_engine_init_seconds,
            "agent_engine_seconds": self.bundle.agent_engine_init_seconds,
            "manager_warm_tokens": manager_warm.token_count if manager_warm else None,
            "manager_warm_seconds": manager_warm.elapsed_seconds if manager_warm else None,
            "agent_warm_tokens": agent_warm.token_count if agent_warm else None,
            "agent_warm_seconds": agent_warm.elapsed_seconds if agent_warm else None,
        }

    def _telemetry_snapshot(self) -> dict[str, object]:
        status = dict(self.scheduler.status_snapshot())
        status["chat_open"] = bool(getattr(self.bundle.runtime, "_chat_mode", False))
        status["inference"] = self._timing_dict(self._current_inference_timing())
        status["manager"] = self._client_snapshot(self.bundle.manager_client)
        status["agent"] = self._client_snapshot(self.bundle.agent_client)

        timers = {
            timer.task_id: timer
            for timer in self.bundle.system_runtime.task_timer_snapshot()
        }
        tasks: list[dict[str, object]] = []
        for task in self.bundle.system_runtime.task_snapshot():
            timer = timers.get(task.task_id)
            tasks.append(
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "method": task.method,
                    "timer": (
                        None
                        if timer is None
                        else {
                            "period_seconds": timer.period_seconds,
                            "enabled": timer.enabled,
                            "next_fire_monotonic": timer.next_fire_monotonic,
                        }
                    ),
                }
            )
        status["tasks"] = tasks
        return status

    def _current_inference_timing(self) -> InferenceTiming:
        clients = [self.bundle.manager_client, *self.bundle.agent_clients]
        for client in clients:
            timing = client.inference_timing
            if timing.phase != "idle":
                return timing

        finished = [
            client.inference_timing
            for client in clients
            if client.inference_timing.finished_at is not None
        ]
        if finished:
            return max(finished, key=lambda item: item.finished_at or 0.0)
        return self.bundle.manager_client.inference_timing

    @staticmethod
    def _client_snapshot(client) -> dict[str, object]:
        response = client.last_response
        return {
            "resident_tokens": client.resident_tokens,
            "last": (
                None
                if response is None
                else {
                    "cached_tokens": response.cached_tokens,
                    "prompt_evaluated_tokens": response.prompt_evaluated_tokens,
                    "completion_tokens": response.completion_tokens,
                    "prompt_seconds": response.prompt_seconds,
                    "generation_seconds": response.generation_seconds,
                    "elapsed_seconds": response.elapsed_seconds,
                }
            ),
        }

    @staticmethod
    def _timing_dict(timing: InferenceTiming) -> dict[str, object]:
        return {
            "phase": timing.phase,
            "phase_started": timing.phase_started,
            "prefill_seconds": timing.prefill_seconds,
            "generation_seconds": timing.generation_seconds,
            "total_seconds": timing.total_seconds,
            "finished_at": timing.finished_at,
        }

    @staticmethod
    def _safe_send(client: _ClientConnection, payload: dict[str, object]) -> None:
        try:
            client.send(payload)
        except OSError:
            LOGGER.debug("CORE send failed client=%s", client.client_name, exc_info=True)