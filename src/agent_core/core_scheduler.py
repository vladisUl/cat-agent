from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import os
import threading
import time
import uuid

from orchestration.manager import AutonomousTaskCompletion, ManagerTurn
from orchestration.system_events import SystemEvent
from .types import InferenceTiming
from .metrics import current_metrics

LOGGER = logging.getLogger(__name__)
MANAGER_PRIORITY = 0
HARDWARE_EVENT_PRIORITY = 10
DEFAULT_EVENT_PRIORITY = 100


class QueueFull(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _ManagerQueryResult:
    task_id: int
    result: str


@dataclass(slots=True)
class _PriorityRequest:
    kind: str
    label: str
    payload: str | SystemEvent | _ManagerQueryResult
    queued_at: float
    priority: int
    coalesce_key: str | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    iterator: object = None
    context: object = None
    started_at: float | None = None
    cancelled: bool = False
    busy_seconds: float = 0.0
    metrics: dict = field(default_factory=dict)


class CoreScheduler:
    """One executor, cooperative steps, bounded admission and explicit completion.

    A model call or running process is not interrupted by a higher priority input.
    All runtime/KV mutation, including iterator cleanup, runs on the executor.
    """

    def __init__(self, bundle, *, on_human_turn=None, on_completed=None,
                 on_notification=None, on_status=None, on_model_event=None,
                 poll_interval=0.02, max_pending=None):
        self.bundle = bundle
        self.on_human_turn = on_human_turn
        self.on_completed = on_completed
        self.on_notification = on_notification
        self.on_status = on_status
        self.on_model_event = on_model_event
        self.poll_interval = poll_interval
        self.max_pending = int(max_pending or os.getenv("CAT_AGENT_MAX_PENDING", "128"))
        if self.max_pending < 2:
            raise ValueError("CAT_AGENT_MAX_PENDING must be >= 2")
        self._queue_lock = threading.RLock()
        self._pending = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cat-agent")
        self._active_request = None
        self._active_future = None
        self._active_started = None
        self._last_request_seconds = None
        self._last_metrics = {}
        self._contexts = {}
        self._released_sessions = set()
        self._stop = threading.Event()
        self._thread = None
        self._rejected = 0

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self.bundle.manager_client.set_event_handler(self._model_event)
        for client in getattr(self.bundle, "agent_clients", ()):
            client.set_event_handler(self._model_event)
        self._thread = threading.Thread(target=self._run, name="cat-agent-scheduler", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        if self._active_future is not None:
            self._active_future.result()
            self._poll_future()
        with self._queue_lock:
            remaining, self._pending = self._pending, []
        for item in remaining:
            self._executor.submit(self._discard, item).result()
            self._complete(item, ManagerTurn("cancelled", "CORE остановлен"))
        self._executor.submit(self._close_all_contexts).result()
        self.bundle.manager_client.set_event_handler(None)
        self._executor.shutdown(wait=True, cancel_futures=False)

    def _close_all_contexts(self):
        for key in list(self._contexts):
            self._close_context(key)

    def submit_user(self, text, *, session_id="", request_id=None):
        return self._submit_input(text, "user", MANAGER_PRIORITY, session_id, request_id)

    def _submit_input(self, text, label, priority, session_id, request_id):
        if not text.strip():
            raise ValueError("user text must be non-empty")
        item = _PriorityRequest("user", label, text.strip(), time.monotonic(), priority,
                                session_id=session_id)
        if request_id is not None:
            item.request_id = request_id
        self._admit(item)
        return item.request_id

    def _append_locked(self, item):
        # Reserve admission space for people even during an event burst.
        limit = self.max_pending if item.kind == "user" else max(1, self.max_pending - 8)
        if len(self._pending) + int(self._active_request is not None) >= limit:
            self._rejected += 1
            LOGGER.error("CORE queue full rejected=%s label=%s", self._rejected, item.label)
            raise QueueFull("CORE очередь заполнена; запрос не принят")
        self._pending.append(item)

    def _admit(self, item):
        with self._queue_lock:
            self._append_locked(item)
        self._emit_status()

    def release_human_session(self, session_id=""):
        with self._queue_lock:
            for item in [*self._pending, self._active_request]:
                if item is not None and item.kind == "user" and item.label != "voice" and item.session_id == session_id:
                    item.cancelled = True
            self._released_sessions.add(session_id)
        # Serialized with model work; a newer client's context is a different key.
        self._executor.submit(self._release_context, session_id)

    def _release_context(self, session_id):
        key = "human:" + session_id
        context = self._contexts.get(key)
        if context is not None:
            context.human_session_released()
        with self._queue_lock:
            live = any(item is not None and item.session_id == session_id
                       for item in [*self._pending, self._active_request])
        if not live:
            self._close_context(key)
            self._released_sessions.discard(session_id)

    def enqueue_external_event(self, event, *, priority=HARDWARE_EVENT_PRIORITY, coalesce=False):
        self._validate_external_priority(priority)
        self._enqueue_event(event, priority=priority,
                            coalesce_key=self._event_label(event) if coalesce else None)

    def _enqueue_event(self, event, *, priority, coalesce_key):
        with self._queue_lock:
            if coalesce_key is not None and self._has_coalesced_event(coalesce_key):
                return
            self._append_locked(_PriorityRequest("system", self._event_label(event), event,
                                                 event.created_monotonic, priority, coalesce_key))
        self._emit_status()

    def _has_coalesced_event(self, key):
        return any(item is not None and item.coalesce_key == key
                   for item in [*self._pending, self._active_request])

    def _poll_system_events(self):
        for event in self.bundle.system_runtime.poll_due(busy=False):
            try:
                self._enqueue_event(event, priority=DEFAULT_EVENT_PRIORITY,
                                    coalesce_key=self._event_label(event) if event.source == "timer" else None)
            except QueueFull:
                LOGGER.error("CORE timer activation rejected task=%s", event.task_id)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._poll_system_events()
                self._poll_future()
                self._start_next()
            except Exception:
                LOGGER.exception("CORE scheduler loop failed")
            self._stop.wait(self.poll_interval)

    def _runnable(self, item):
        if not item.cancelled and getattr(item.context, "_waiting_for_worker", False) and self.bundle.runtime.pool.acquire() is None:
            return False
        if item.cancelled or item.iterator is not None:
            return True
        if item.kind == "system" and item.payload.task_id is not None:
            can_begin = getattr(self.bundle.runtime, "can_begin_autonomous_task", None)
            return can_begin is None or can_begin(item.payload)
        if item.kind in {"user", "manager"}:
            # A later input in the same dialogue cannot overtake its suspended turn.
            return not any(other is not item and other.kind in {"user", "manager"} and
                           self._context_key(other) == self._context_key(item) and
                           other.iterator is not None for other in self._pending)
        return True

    def _start_next(self):
        if self._active_future is not None:
            return
        with self._queue_lock:
            candidates = [i for i, item in enumerate(self._pending) if self._runnable(item)]
            if not candidates:
                return
            index = min(candidates, key=lambda i: (self._pending[i].priority, self._pending[i].queued_at, i))
            item = self._pending.pop(index)
            self._active_request = item
        self._active_started = time.monotonic()
        if item.started_at is None:
            item.started_at = self._active_started
        self._active_future = self._executor.submit(self._advance, item)
        self._emit_status()

    def _advance(self, item):
        token = current_metrics.set(item.metrics)
        try:
            if item.cancelled:
                self._discard(item)
                return ManagerTurn("cancelled", "")
            if item.iterator is None:
                item.iterator = self._steps(item)
            try:
                next(item.iterator)
                return None
            except StopIteration as end:
                return end.value or ManagerTurn("silent", "")
        except Exception as exc:
            LOGGER.exception("CORE request failed id=%s", item.request_id)
            self._discard(item)
            return ManagerTurn("error", str(exc))
        finally:
            current_metrics.reset(token)

    @staticmethod
    def _context_key(item):
        if item.label == "voice":
            return "voice"
        if item.kind == "user":
            return "human:" + item.session_id
        return "notifications"

    def _context(self, item):
        key = self._context_key(item)
        if key not in self._contexts:
            fork = getattr(self.bundle.runtime, "fork_context", None)
            self._contexts[key] = fork(key) if fork is not None else self.bundle.runtime
            if fork is not None:
                self._contexts[key].client.set_event_handler(self._model_event)
        item.context = self._contexts[key]
        return item.context

    def _steps(self, item):
        if item.kind == "system" and item.payload.task_id is not None:
            runtime = self.bundle.runtime
            result = runtime.begin_autonomous_task(item.payload)
            if isinstance(result, AutonomousTaskCompletion):
                return result
            try:
                while True:
                    completion = runtime.step_autonomous_task(result)
                    if completion is not None:
                        return completion
                    yield
            finally:
                if result.worker.state.value != "FREE":
                    result.worker.sleep_to_base()
        runtime = self._context(item)
        if item.kind == "user":
            steps = getattr(runtime, "user_message_steps", None)
            if steps is not None:
                return (yield from steps(item.payload))
            return runtime.user_message(item.payload)
        if item.kind == "manager":
            steps = getattr(runtime, "query_result_steps", None)
            if steps is not None:
                return (yield from steps(item.payload.task_id, item.payload.result))
            return runtime.autonomous_query_result(item.payload.task_id, item.payload.result)
        return runtime.system_event(item.payload)

    def _discard(self, item):
        if item.iterator is not None:
            item.iterator.close()
        if item.context is not None:
            item.context._abort_context()

    def _poll_future(self):
        if self._active_future is None or not self._active_future.done():
            return
        self._finish_regular_request(self._active_future)

    def _finish_regular_request(self, future):
        item = self._active_request
        try:
            result = future.result()
        except Exception as exc:
            result = ManagerTurn("error", str(exc))
        if item is None:
            return
        now = time.monotonic()
        item.busy_seconds += now - (self._active_started or now)
        if result is None:
            with self._queue_lock:
                self._pending.append(item)
        else:
            if isinstance(result, AutonomousTaskCompletion):
                if result.query_task_id is not None:
                    # The activation already occupied one admission slot.
                    item.kind = "manager"
                    item.label = f"query-result:task:{result.query_task_id}"
                    item.payload = _ManagerQueryResult(result.query_task_id, result.query_result)
                    item.priority = MANAGER_PRIORITY
                    item.iterator = None
                    with self._queue_lock:
                        self._pending.append(item)
                    result = None
                else:
                    result = result.turn or ManagerTurn("silent", "")
            if result is not None:
                self._complete(item, result)
        self._active_request = None
        self._active_future = None
        self._active_started = None
        if item.session_id in self._released_sessions:
            self._executor.submit(self._release_context, item.session_id)
        self._emit_status()

    def _complete(self, item, turn):
        self._last_request_seconds = time.monotonic() - item.queued_at
        self._last_metrics = {**item.metrics, "request_id": item.request_id,
                              "total_seconds": self._last_request_seconds,
                              "queue_seconds": max(0, self._last_request_seconds - item.busy_seconds),
                              "busy_seconds": item.busy_seconds}
        LOGGER.info("CORE complete id=%s session=%s kind=%s metrics=%s", item.request_id, item.session_id, turn.kind, self._last_metrics)
        try:
            if item.kind == "user":
                if self.on_completed is not None:
                    self.on_completed(item, turn)
                elif turn.kind != "silent" and self.on_human_turn is not None:
                    self.on_human_turn(turn)
            elif turn.kind == "reply" and turn.text and self.on_notification is not None:
                self.on_notification(turn)
        except Exception:
            LOGGER.exception("CORE completion delivery failed id=%s", item.request_id)
        if item.kind == "user" and item.label != "voice" and item.session_id in self._released_sessions:
            self._executor.submit(self._close_context, self._context_key(item))


    def _close_context(self, key):
        context = self._contexts.pop(key, None)
        if context is not None and context is not self.bundle.runtime:
            context.client.close()

    def active_request_label(self):
        return self._active_request.label if self._active_request else ""

    def status_snapshot(self):
        item = self._active_request
        client = getattr(item.context, "client", self.bundle.manager_client) if item else self.bundle.manager_client
        with self._queue_lock:
            pending = len(self._pending)
            background = any(r.iterator is not None for r in self._pending)
            highest = min((r.priority for r in self._pending), default=None)
        inference_client = next(
            (candidate for candidate in [client, *getattr(self.bundle, "agent_clients", ())]
             if candidate.inference_timing.phase != "idle"), client,
        )
        return {"state": "BUSY" if item else "IDLE", "label": item.label if item else "",
                "priority": item.priority if item else None, "request_id": item.request_id if item else None,
                "request_started_monotonic": self._active_started,
                "last_request_seconds": self._last_request_seconds, "last_metrics": self._last_metrics,
                "pending": pending, "background": background, "rejected": self._rejected,
                "waiting_priority": highest, "manager_resident_tokens": client.resident_tokens,
                "agent_resident_tokens": self.bundle.agent_client.resident_tokens,
                "inference": self._timing_dict(inference_client.inference_timing)}

    def _model_event(self, label, event, payload):
        if self.on_model_event is not None:
            item = self._active_request
            client = getattr(item.context, "client", self.bundle.manager_client) if item else self.bundle.manager_client
            if label != "manager":
                client = next((c for c in getattr(self.bundle, "agent_clients", ()) if c.label == label), client)
            self.on_model_event(label, event, payload, client.inference_timing)
        self._emit_status()

    def _emit_status(self):
        if self.on_status is not None:
            try:
                self.on_status(self.status_snapshot())
            except Exception:
                LOGGER.debug("CORE status callback failed", exc_info=True)

    def _take_next_request(self):
        with self._queue_lock:
            if not self._pending:
                return None
            return self._pending.pop(min(range(len(self._pending)), key=lambda i: (self._pending[i].priority, self._pending[i].queued_at, i)))

    @staticmethod
    def _event_label(event):
        return f"{event.source}:{event.name}"

    @staticmethod
    def _validate_external_priority(priority):
        if priority <= MANAGER_PRIORITY:
            raise ValueError("external event priority must be > 0")

    @staticmethod
    def _timing_dict(timing):
        return {name: getattr(timing, name) for name in ("phase", "phase_started", "prefill_seconds", "generation_seconds", "total_seconds", "finished_at")}
