from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import socket
import threading
import uuid
from .connection import ClientConnection as _ClientConnection
from .outbox import NotificationOutbox
from .firebase_push import FirebasePushSender, read_firebase_tokens
from .socket_owner import SocketOwner
from typing import Any

from orchestration.manager import ManagerTurn
from orchestration.tasks import TaskStoreError

from .core_scheduler import CoreScheduler, HARDWARE_EVENT_PRIORITY
from .types import InferenceTiming
from .voice_scheduler import (
    VOICE_PRIORITY,
    VOICE_REQUEST_LABEL,
    VoiceCoreScheduler,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CORE_SOCKET = Path("/run/cat-agent/core.sock")


class CoreServer:
    """Local IPC boundary around the long-lived model/runtime core."""

    def __init__(
        self,
        bundle,
        *,
        path: Path | None = None,
        scheduler: CoreScheduler | None = None,
        outbox_path: Path | None = None,
    ) -> None:
        env_path = os.getenv("CAT_AGENT_CORE_SOCKET", "").strip()
        self.path = Path(env_path) if env_path else (path or DEFAULT_CORE_SOCKET)
        self.bundle = bundle
        self.outbox = NotificationOutbox(outbox_path or Path(os.getenv("CAT_AGENT_OUTBOX", "/var/lib/cat-agent/outbox.sqlite3")), self._send_notification)
        self.scheduler = scheduler or VoiceCoreScheduler(
            bundle,
            on_completed=self._deliver_completed,
            on_notification=self._deliver_notification,
            on_status=self._deliver_status,
            on_model_event=self._deliver_model_event,
        )

        self._lock = threading.RLock()
        self._firebase_sender = FirebasePushSender()
        self._human: _ClientConnection | None = None
        self._voice_turn: _ClientConnection | None = None
        self._voice_request_id: str | None = None
        self._fallback: _ClientConnection | None = None
        self._clients: list[_ClientConnection] = []
        self._server_socket: socket.socket | None = None
        self._stop = threading.Event()
        self._socket_owner = SocketOwner(self.path)

    def start(self) -> None:
        if self._server_socket is not None:
            return
        self._socket_owner.acquire()
        sock = None
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(self.path))
            self._socket_owner.bound()
            os.chmod(self.path, 0o660)
            sock.listen(16)
            sock.settimeout(0.25)
        except Exception:
            if sock is not None:
                sock.close()
            self._socket_owner.release()
            raise

        self._stop.clear()
        self._server_socket = sock
        try:
            self.outbox.start()
            self.scheduler.start()
        except Exception:
            self.close()
            raise
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
            connection.start()
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
            self._voice_turn = None
            self._fallback = None
        for client in clients:
            client.close()

        try:
            self.scheduler.close()
        finally:
            try:
                self.outbox.close()
            finally:
                self._socket_owner.release()
        LOGGER.info("CORE socket stopped: %s", self.path)

    def session_owner(self) -> str | None:
        with self._lock:
            return self._human.client_name if self._human is not None else None

    def _client_loop(self, client: _ClientConnection) -> None:
        LOGGER.info("CORE client connected")
        try:
            reader = client.sock.makefile("r", encoding="utf-8", newline="\n")
            try:
                while True:
                    raw = reader.readline(65537)
                    if not raw:
                        break
                    if len(raw) > 65536:
                        raise ValueError("IPC message exceeds 64 KiB")
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
        except (OSError, ValueError):
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
            if self._release_human(client):
                self.scheduler.release_human_session(client.session_id)
            self._safe_send(client, {"type": "released"})
            return

        if message_type == "register_fallback":
            self._register_fallback(client, str(item.get("client", "telegram")))
            return

        if message_type == "voice":
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError("voice text must be non-empty")
            self._submit_voice_turn(
                client,
                text,
                str(item.get("client", "voice")),
            )
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
            with self._lock:
                if self._human is not client:
                    return
                request_id = self.scheduler.submit_user(text, session_id=client.session_id)
                self._safe_send(client, {"type": "accepted", "request_id": request_id, "session_id": client.session_id})
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

        if message_type == "notification_ack":
            if self._human is client:
                self.outbox.acknowledge(str(item.get("notification_id", "")))
            return

        if message_type == "warm_kv2":
            with self._lock:
                owner = self._human is client
                current = self._human.client_name if self._human is not None else None
            if not owner:
                self._safe_send(
                    client,
                    {"type": "busy", "owner": current, "text": "Гена занят"},
                )
                return
            state = self.scheduler.request_manager_kv2_warm()
            self._safe_send(client, {"type": "kv2_warm", "state": state})
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
                    "session_id": client.session_id,
                    "core": self._core_info(),
                    "status": self.scheduler.status_snapshot(),
                },
            )
        else:
            LOGGER.info("CORE human session busy owner=%s requested_by=%s", owner, client_name)
            self._safe_send(
                client,
                {"type": "busy", "owner": owner, "text": "Гена занят"},
            )

    def _release_human(self, client: _ClientConnection) -> bool:
        released = False
        with self._lock:
            if self._human is client:
                LOGGER.info("CORE human session released by %s", client.client_name)
                self._human = None
                released = True
            client.human_owner = False
        return released

    def _submit_voice_turn(self, client, text, name):
        with self._lock:
            if self._voice_request_id is not None:
                self._safe_send(client, {"type": "busy", "text": "Голосовой запрос уже выполняется"})
                return
            request_id = uuid.uuid4().hex
            self._voice_turn = client
            self._voice_request_id = request_id
            client.client_name = name.strip() or "voice"
            try:
                self.scheduler.submit_voice(text, session_id=client.session_id, request_id=request_id)
            except Exception:
                self._voice_turn = None
                self._voice_request_id = None
                raise
            self._safe_send(client, {"type": "voice_accepted", "request_id": request_id,
                                     "session_id": client.session_id, "client": client.client_name,
                                     "priority": VOICE_PRIORITY})

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
        released_human = False
        with self._lock:
            if self._human is client:
                LOGGER.info("CORE human session auto-release client=%s", client.client_name)
                self._human = None
                released_human = True
            if self._voice_turn is client:
                LOGGER.info(
                    "CORE voice client disconnected while turn is pending client=%s",
                    client.client_name,
                )
            if self._fallback is client:
                LOGGER.info("CORE fallback channel disconnected client=%s", client.client_name)
                self._fallback = None
            try:
                self._clients.remove(client)
            except ValueError:
                pass
            client.human_owner = False
            client.fallback = False
        if released_human:
            self.scheduler.release_human_session(client.session_id)

    def _deliver_completed(self, request, turn):
        chat_open = bool(getattr(request.context, "_chat_mode", False))
        with self._lock:
            if request.label == VOICE_REQUEST_LABEL:
                if self._voice_request_id != request.request_id:
                    return
                target = self._voice_turn
                self._voice_turn = None
                self._voice_request_id = None
            else:
                target = self._human
            if target is None or target.session_id != request.session_id:
                LOGGER.info("CORE result has no originating client id=%s", request.request_id)
                return
            payload = {"request_id": request.request_id, "session_id": request.session_id,
                       "kind": turn.kind, "text": turn.text, "chat_open": chat_open}
            if turn.kind not in {"silent", "cancelled"}:
                self._safe_send(target, {"type": "reply", **payload})
            self._safe_send(target, {"type": "completed", **payload})

    def _deliver_human_turn(self, turn: ManagerTurn) -> None:
        route = self._active_request_label()
        if route == VOICE_REQUEST_LABEL:
            with self._lock:
                target = self._voice_turn
                self._voice_turn = None
            if target is None:
                LOGGER.warning("CORE voice reply has no voice client: %r", turn.text)
                return
            self._safe_send(
                target,
                {"type": "reply", "kind": turn.kind, "text": turn.text},
            )
            return

        with self._lock:
            target = self._human
        if target is None:
            LOGGER.warning("CORE human reply has no active human client: %r", turn.text)
            return
        self._safe_send(
            target,
            {"type": "reply", "kind": turn.kind, "text": turn.text},
        )

    def _deliver_notification(self, turn):
        self.outbox.enqueue(turn.text)

    def _send_notification(self, identifier, text):
        with self._lock:
            target = self._human
        if target is not None:
            if target.send({"type": "notification", "notification_id": identifier,
                            "kind": "reply", "text": text}):
                return False  # Client acknowledges reception through IPC.
        self._send_firebase_push(text)
        return True  # Provider acceptance, not confirmation of display on phone.

    def _send_firebase_push(self, text: str) -> None:
        self._firebase_sender.send(text)

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
        route = self._active_request_label()
        with self._lock:
            if route == VOICE_REQUEST_LABEL:
                target = self._voice_turn
            else:
                target = self._human
        request = getattr(self.scheduler, "_active_request", None)
        if target is None or (request is not None and (request.kind != "user" or target.session_id != request.session_id)):
            return
        self._safe_send(
            target,
            {
                "request_id": request.request_id if request is not None else None,
                "type": "model_event",
                "label": label,
                "event": event,
                "payload": payload,
                "timing": self._timing_dict(timing),
            },
        )

    def _active_request_label(self) -> str:
        active_request_label = getattr(self.scheduler, "active_request_label", None)
        if callable(active_request_label):
            return str(active_request_label())
        return str(self.scheduler.status_snapshot().get("label", ""))

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
        with self._lock:
            human = self._human
        contexts = getattr(self.scheduler, "_contexts", {})
        context = contexts.get("human:" + human.session_id, self.bundle.runtime) if human else self.bundle.runtime
        status["chat_open"] = bool(getattr(context, "_chat_mode", False))
        status["inference"] = self._timing_dict(self._current_inference_timing())
        status["manager"] = self._client_snapshot(getattr(context, "client", self.bundle.manager_client))
        status["agent"] = self._client_snapshot(self.bundle.agent_client)

        try:
            timers = {timer.task_id: timer for timer in self.bundle.system_runtime.task_timer_snapshot()}
            task_snapshot = self.bundle.system_runtime.task_snapshot()
        except TaskStoreError as exc:
            status["task_system_error"] = str(exc)
            timers, task_snapshot = {}, ()
        tasks: list[dict[str, object]] = []
        for task in task_snapshot:
            timer = timers.get(task.task_id)
            tasks.append(
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "executor": task.executor,
                    "method": task.method,
                    "enabled": task.enabled,
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
        clients = [self.bundle.manager_client, *self.bundle.agent_clients,
                   *(c.client for c in list(getattr(self.scheduler, "_contexts", {}).values()))]
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

