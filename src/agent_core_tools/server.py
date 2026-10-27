"""Independent native-tools CORE IPC: optional serialized Conversation manager.

No Task SYSTEM, legacy scheduler, or agent imports. Tool calls are native.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math
import time
import json
import logging
import os
from pathlib import Path
import socket
import threading
import uuid
from typing import Any

from .socket_owner import SocketOwner
from .priority_scheduler import PriorityScheduler, QueueFullError
from orchestration_tools.contracts import ManagerRuntime


LOGGER = logging.getLogger(__name__)
DEFAULT_SOCKET = Path("/run/cat-agent/litert-tools.sock")
MAX_FRAME_BYTES = 65536
MAX_CLIENTS = 16
MAX_PENDING = 16
MAX_BACKGROUND_STEPS = 8
MAX_BACKGROUND_STEP_CHARS = 2048
MAX_FINISHED_JOBS = 32
MAX_HARDWARE_SOURCE_CHARS = 64
MAX_HARDWARE_TEXT_CHARS = 2048
MAX_BACKGROUND_RUNTIME = 3600.0


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


@dataclass
class _BackgroundJob:
    job_id: str
    client: _Client
    epoch: int
    agent_id: str
    steps: tuple[str, ...]
    index: int = 0
    cancelled: bool = False
    state: str = "QUEUED"
    max_runtime_seconds: float = 120.0
    started_at: float = field(default_factory=time.monotonic)
    last_result: str = ""
    delivery_failed: bool = False

    @property
    def deadline(self) -> float:
        return self.started_at + self.max_runtime_seconds

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.job_id, "agent_id": self.agent_id,
            "state": self.state, "completed_steps": self.index,
            "total_steps": len(self.steps), "cancelled": self.cancelled,
            "last_result": self.last_result,
        }


class CoreServer:
    """Dedicated IPC and a cooperative, single-engine priority scheduler."""

    def __init__(
        self, path: Path = DEFAULT_SOCKET, *,
        manager: ManagerRuntime | None = None,
        background_enabled: bool = False,
        hardware_probe_enabled: bool = False,
        task_worker_socket: Path | None = None,
        task_agent_permissions: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.path = Path(path)
        self._socket_owner = SocketOwner(self.path)
        self._socket: socket.socket | None = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._close_lock = threading.RLock()
        self._clients: dict[_Client, threading.Thread] = {}
        self._human: _Client | None = None
        self._manager = manager
        self._scheduler = PriorityScheduler(max_pending=MAX_PENDING) if manager else None
        self._background_enabled = background_enabled
        self._hardware_probe_enabled = hardware_probe_enabled
        self._backgrounds: dict[str, _BackgroundJob] = {}
        self._finished_backgrounds: OrderedDict[str, tuple[_Client, int, dict]] = OrderedDict()
        self._task_worker = None
        if task_worker_socket is not None:
            if manager is None:
                raise ValueError("TASK worker requires native manager")
            from .task_worker import NativeTaskWorker
            from task_system.client import TaskRPC
            self._task_worker = NativeTaskWorker(
                self._scheduler, manager, rpc=TaskRPC(task_worker_socket),
                agent_permissions=task_agent_permissions,
                agent_blocked=self._task_agent_blocked,
                stop_event=self._stop,
            )

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
        if self._task_worker is not None:
            try:
                self._task_worker.start()
            except Exception:
                self.close()
                raise
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
                    self._cancel_backgrounds(client)
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
            status = {"ready": True, **self._scheduler.snapshot()}
            status["background_jobs"] = [
                job.snapshot() for job in self._backgrounds.values()
            ]
            if hasattr(self._manager, "agent_status"):
                status["agents"] = self._manager.agent_status()
            if self._task_worker is not None:
                status["task_worker"] = self._task_worker.snapshot()
            return status

    def _task_agent_blocked(self, agent_id: str) -> bool:
        # Worker calls this on its RPC thread; model callback holds the same
        # lock for status changes. A waiting background KV is not disposable.
        with self._lock:
            return any(job.agent_id == agent_id for job in self._backgrounds.values())

    def _task_agent_reserved(self, agent_id: str) -> bool:
        return (self._task_worker is not None and
                self._task_worker.is_reserved(agent_id))

    def _schedule_release(self, session_id: str) -> None:
        # Called while holding _lock; serialized AFTER in-flight inference.
        if self._scheduler is not None and not self._stop.is_set():
            self._scheduler.submit(
                "control", lambda: self._manager.release_session(session_id),
            )

    def _client_still_active(self, client: _Client, epoch: int) -> bool:
        with self._lock:
            return (not self._stop.is_set() and self._human is client
                    and client.epoch == epoch)

    def _model_worker(
        self, client: _Client, epoch: int, request_id: str, text: str,
        agent_id: str | None = None,
    ) -> None:
        with self._lock:
            if self._human is not client or client.epoch != epoch or self._stop.is_set():
                return
        LOGGER.info("native-scheduler HUMAN START id=%s agent=%s",
                    request_id, agent_id)
        try:
            try:
                if agent_id is None:
                    guarded = getattr(self._manager, "respond_guarded", None)
                    if callable(guarded):
                        turn = guarded(
                            client.session_id, text,
                            lambda: self._client_still_active(client, epoch),
                        )
                    else:
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
            LOGGER.info("native-scheduler HUMAN FINISH id=%s agent=%s",
                        request_id, agent_id)

    def _finish_background(self, job: _BackgroundJob, state: str, result: str = "") -> None:
        """Move an active job into a bounded per-session terminal ledger.

        Only called under _lock. Never replay a completed native operation.
        """
        job.state = state
        job.last_result = result[:2048]
        self._backgrounds.pop(job.job_id, None)
        self._finished_backgrounds[job.job_id] = (
            job.client, job.epoch, job.snapshot(),
        )
        self._finished_backgrounds.move_to_end(job.job_id)
        while len(self._finished_backgrounds) > MAX_FINISHED_JOBS:
            self._finished_backgrounds.popitem(last=False)

    def _background_status(self, client: _Client, job_id: str) -> dict | None:
        with self._lock:
            job = self._backgrounds.get(job_id)
            if job is not None:
                if job.client is client and job.epoch == client.epoch:
                    return job.snapshot()
                return None
            finished = self._finished_backgrounds.get(job_id)
            if finished is None:
                return None
            owner, epoch, record = finished
            return dict(record) if owner is client and epoch == client.epoch else None

    def _cancel_backgrounds(self, client: _Client, job_id: str | None = None) -> list[str]:
        """Mark work cancelled; model-side context cleanup runs on scheduler."""
        cancelled = []
        for job in self._backgrounds.values():
            if job.client is not client or job.cancelled:
                continue
            if job_id is not None and job.job_id != job_id:
                continue
            job.cancelled = True
            job.state = "CANCELLING"
            cancelled.append(job.job_id)
            if self._scheduler is not None and not self._stop.is_set():
                self._scheduler.submit(
                    "control",
                    lambda agent_id=job.agent_id, jid=job.job_id:
                        self._clear_background_context(agent_id, jid),
                )
        return cancelled

    def _notify_background_terminal(
        self, job: _BackgroundJob, *, ok: bool, result: str,
    ) -> None:
        # A failed write means the transport cannot be trusted. Preserve the
        # terminal state for diagnostics, but never write to that client again.
        if job.delivery_failed:
            return
        self._send_background_event(job, {
            "type": "background_completed",
            "background_id": job.job_id, "agent_id": job.agent_id,
            "completed_steps": job.index, "total": len(job.steps),
            "ok": ok, "state": job.state, "text": result[:2048],
        })

    def _clear_background_context(self, agent_id: str, job_id: str) -> None:
        reset_error = None
        try:
            self._manager.abort_agent_slices(agent_id)
        except Exception as exc:
            reset_error = str(exc)
            LOGGER.exception("Background context cleanup failed agent=%s", agent_id)
        finally:
            with self._lock:
                job = self._backgrounds.get(job_id)
                if job is not None and job.cancelled:
                    state = "FAILED" if reset_error is not None else "CANCELLED"
                    result = reset_error or "Cancelled before next native call"
                    self._finish_background(job, state, result)
                    self._notify_background_terminal(job, ok=False, result=result)

    def _clear_timed_out_context(self, agent_id: str, job_id: str) -> None:
        reset_error = None
        try:
            self._manager.abort_agent_slices(agent_id)
        except Exception as exc:
            reset_error = str(exc)
            LOGGER.exception("Background timeout context reset failed agent=%s", agent_id)
        finally:
            with self._lock:
                job = self._backgrounds.get(job_id)
                if job is not None:
                    state = "FAILED" if reset_error is not None else "TIMED_OUT"
                    result = reset_error or "Deadline reached between native calls"
                    self._finish_background(job, state, result)
                    self._notify_background_terminal(job, ok=False, result=result)

    def _send_background_event(self, job: _BackgroundJob, payload: dict) -> bool:
        """Best-effort IPC: a vanished reader must cancel future model steps.

        Caller holds _lock. The active native turn is never retried.
        """
        if job.delivery_failed:
            return False
        if self._human is not job.client or job.client.epoch != job.epoch:
            self._cancel_backgrounds(job.client, job.job_id)
            return False
        try:
            job.client.send(payload)
            return True
        except OSError as exc:
            LOGGER.info(
                "native-background CLIENT-GONE id=%s error=%s; cancelling remaining steps",
                job.job_id, exc,
            )
            job.delivery_failed = True
            self._cancel_backgrounds(job.client, job.job_id)
            return False

    def _background_step(self, job_id: str) -> None:
        """Run exactly ONE native call, then yield the Engine to priority queue."""
        with self._lock:
            job = self._backgrounds.get(job_id)
            if job is None or job.cancelled or self._stop.is_set():
                return
            position = job.index
            instruction = job.steps[position]
            final = position == len(job.steps) - 1
            if time.monotonic() >= job.deadline:
                LOGGER.info("native-background DEADLINE id=%s before step=%d",
                            job_id, position + 1)
                job.state = "CANCELLING"
                job.cancelled = True
                self._scheduler.submit(
                    "control",
                    lambda agent_id=job.agent_id, jid=job.job_id:
                        self._clear_timed_out_context(agent_id, jid),
                )
                return
            job.state = "RUNNING"
        LOGGER.info("native-background STEP id=%s agent=%s step=%d/%d",
                    job_id, job.agent_id, position + 1, len(job.steps))
        try:
            from orchestration_tools.agents import AgentState
            outcome = self._manager.run_agent_slice(
                job.agent_id, instruction, final=final,
            )
            ok = outcome.state in (AgentState.FREE, AgentState.WAITING)
            answer = outcome.text[:2048]
        except Exception as exc:
            LOGGER.exception("native-background ERROR id=%s", job_id)
            ok, answer = False, str(exc)[:2048]
        with self._lock:
            if job.cancelled or self._stop.is_set():
                return
            job.index += 1
            job.last_result = answer
            if not self._send_background_event(job, {
                "type": "background_progress",
                "background_id": job_id, "agent_id": job.agent_id,
                "step": job.index, "total": len(job.steps),
                "ok": ok, "text": answer,
            }):
                return
            if ok and not final and time.monotonic() >= job.deadline:
                job.cancelled = True
                job.state = "CANCELLING"
                self._scheduler.submit(
                    "control",
                    lambda agent_id=job.agent_id, jid=job.job_id:
                        self._clear_timed_out_context(agent_id, jid),
                )
                return
            if ok and not final:
                job.state = "WAITING"
                try:
                    self._scheduler.submit(
                        "background", lambda: self._background_step(job_id),
                        continuation=True,
                    )
                    return
                except (QueueFullError, RuntimeError) as exc:
                    # No replay. Reset suspended agent on the model worker.
                    ok, answer = False, f"Failed to queue next slice: {exc}"
                    self._scheduler.submit(
                        "control",
                        lambda: self._manager.abort_agent_slices(job.agent_id),
                    )
            self._finish_background(job, "COMPLETED" if ok else "FAILED", answer)
            self._notify_background_terminal(job, ok=ok, result=answer)
        LOGGER.info("native-background FINISH id=%s completed=%s/%s ok=%s",
                    job_id, job.index, len(job.steps), ok)

    def _submit_background(self, client: _Client, message: dict[str, Any]) -> None:
        if not self._background_enabled or self._manager is None:
            client.send({"type": "error", "error": "background probe disabled"})
            return
        agent_id = message.get("agent")
        steps = message.get("steps")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("background_task requires agent")
        if not isinstance(steps, list) or not 2 <= len(steps) <= MAX_BACKGROUND_STEPS:
            raise ValueError("background_task requires 2..8 steps")
        if not all(isinstance(t, str) and 0 < len(t.strip()) <= MAX_BACKGROUND_STEP_CHARS for t in steps):
            raise ValueError("invalid background step")
        for text in steps:
            text.encode("utf-8", "strict")
        requested_runtime = message.get("max_runtime_seconds", 120.0)
        if (isinstance(requested_runtime, bool) or
                not isinstance(requested_runtime, (int, float)) or
                not math.isfinite(requested_runtime) or
                not 1.0 <= requested_runtime <= MAX_BACKGROUND_RUNTIME):
            raise ValueError("max_runtime_seconds must be 1..3600")
        with self._lock:
            if self._human is not client:
                client.send({"type": "busy",
                             "owner": self._human.name if self._human else None,
                             "text": "Гена занят"})
                return
            if not any(a["agent_id"] == agent_id for a in self._manager.agent_status()):
                raise ValueError(f"agent is not configured: {agent_id}")
            if any(j.agent_id == agent_id for j in self._backgrounds.values()):
                client.send({"type": "error", "error": "agent already has a background job"})
                return
            if self._task_agent_reserved(agent_id):
                client.send({"type": "error", "error": "agent reserved for TASK SYSTEM"})
                return
            job_id = uuid.uuid4().hex
            job = _BackgroundJob(job_id, client, client.epoch, agent_id,
                                 tuple(t.strip() for t in steps),
                                 max_runtime_seconds=float(requested_runtime))
            try:
                self._scheduler.submit(
                    "background", lambda: self._background_step(job_id),
                )
            except (QueueFullError, RuntimeError) as exc:
                client.send({"type": "error", "error": str(exc)})
                return
            self._backgrounds[job_id] = job
            client.send({
                "type": "background_accepted", "background_id": job_id,
                "agent_id": agent_id, "total": len(steps),
            })

    def _hardware_probe_worker(
        self, client: _Client, epoch: int, event_id: str,
        source: str, name: str, agent_id: str, task: str,
    ) -> None:
        """Diagnostic ONLY: one agent turn, never a Task SYSTEM RUN."""
        if not self._client_still_active(client, epoch):
            return
        LOGGER.info("native-hardware START id=%s source=%s name=%s agent=%s",
                    event_id, source, name, agent_id)
        try:
            from orchestration_tools.agents import AgentState
            outcome = self._manager.run_agent(agent_id, task)
            ok = outcome.state is AgentState.FREE
            result = outcome.text[:2048]
        except Exception as exc:
            LOGGER.exception("native-hardware ERROR id=%s", event_id)
            ok, result = False, str(exc)[:2048]
        LOGGER.info("native-hardware FINISH id=%s ok=%s", event_id, ok)
        with self._lock:
            if not self._client_still_active(client, epoch):
                return
            try:
                client.send({
                    "type": "hardware_completed", "event_id": event_id,
                    "source": source, "name": name, "agent_id": agent_id,
                    "ok": ok, "text": result,
                })
            except OSError:
                LOGGER.info("native-hardware CLIENT-GONE id=%s", event_id)

    def _submit_hardware_probe(self, client: _Client, message: dict[str, Any]) -> None:
        if not self._hardware_probe_enabled or self._manager is None:
            client.send({"type": "error", "error": "hardware probe disabled"})
            return
        if set(message) != {"type", "source", "name", "agent", "text"}:
            raise ValueError("hardware_probe accepts only source, name, agent and text")
        source, name = message.get("source"), message.get("name")
        agent_id, task = message.get("agent"), message.get("text")
        if not isinstance(source, str) or not 0 < len(source.strip()) <= MAX_HARDWARE_SOURCE_CHARS:
            raise ValueError("hardware_probe requires a short source")
        if not isinstance(name, str) or not 0 < len(name.strip()) <= MAX_HARDWARE_SOURCE_CHARS:
            raise ValueError("hardware_probe requires a short name")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("hardware_probe requires agent")
        if not isinstance(task, str) or not 0 < len(task.strip()) <= MAX_HARDWARE_TEXT_CHARS:
            raise ValueError("hardware_probe requires short text")
        for value in (source, name, agent_id, task):
            value.encode("utf-8", "strict")
        with self._lock:
            if self._human is not client:
                client.send({
                    "type": "busy",
                    "owner": self._human.name if self._human else None,
                })
                return
            if not any(a["agent_id"] == agent_id for a in self._manager.agent_status()):
                raise ValueError(f"agent is not configured: {agent_id}")
            if any(job.agent_id == agent_id for job in self._backgrounds.values()):
                client.send({"type": "error", "error": "hardware probe agent has a background job"})
                return
            if self._task_agent_reserved(agent_id):
                client.send({"type": "error", "error": "agent reserved for TASK SYSTEM"})
                return
            event_id, epoch = uuid.uuid4().hex, client.epoch
            try:
                self._scheduler.submit(
                    "hardware", lambda: self._hardware_probe_worker(
                        client, epoch, event_id,
                        source.strip().lower(), name.strip(), agent_id, task.strip(),
                    ),
                )
            except (QueueFullError, RuntimeError) as exc:
                client.send({"type": "error", "error": str(exc)})
                return
            client.send({
                "type": "hardware_accepted", "event_id": event_id,
                "source": source.strip().lower(), "name": name.strip(),
                "agent_id": agent_id,
            })

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
                        "stage": (1 if self._manager is None else
                                  5 if self._task_worker is not None else
                                  4 if self._background_enabled else 2),
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
                    self._cancel_backgrounds(client)
                    self._schedule_release(client.session_id)
            client.send({"type": "released"})
            return
        if message_type == "hardware_probe":
            self._submit_hardware_probe(client, message)
            return
        if message_type == "background_status":
            job_id = message.get("background_id")
            if not isinstance(job_id, str) or not job_id.strip():
                raise ValueError("background_status requires background_id")
            with self._lock:
                if self._human is not client:
                    client.send({"type": "busy",
                                 "owner": self._human.name if self._human else None})
                    return
                result = self._background_status(client, job_id)
            client.send({"type": "background_status", "job": result})
            return
        if message_type == "background_task":
            self._submit_background(client, message)
            return
        if message_type == "cancel_background":
            job_id = message.get("background_id")
            if not isinstance(job_id, str) or not job_id.strip():
                raise ValueError("cancel_background requires background_id")
            with self._lock:
                if self._human is not client:
                    client.send({"type": "busy",
                                 "owner": self._human.name if self._human else None})
                    return
                cancelled = self._cancel_backgrounds(client, job_id)
            client.send({
                "type": "background_cancelled", "background_id": job_id,
                "cancelled": bool(cancelled),
            })
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
                    request_id = uuid.uuid4().hex
                    # Capture the admission epoch, not the client's future
                    # epoch: stale requests must never run after release/reacquire.
                    admitted_epoch = client.epoch
                    try:
                        self._scheduler.submit(
                            "human", lambda epoch=admitted_epoch:
                                self._model_worker(
                                    client, epoch, request_id, content, agent_id,
                                ),
                        )
                    except (QueueFullError, RuntimeError) as exc:
                        client.send({"type": "error", "error": str(exc)})
                        return
                    client.send({
                        "type": "accepted", "request_id": request_id,
                        "session_id": client.session_id,
                    })
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
            if self._scheduler is not None:
                self._scheduler.close()
                self._scheduler = None
            if self._task_worker is not None:
                self._task_worker.close()
                self._task_worker = None
            if self._manager is not None:
                self._manager.close()
            self._socket_owner.release()
        LOGGER.info("TOOLS CORE stopped: %s", self.path)
