from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
import threading
import time

from orchestration.manager import (
    AutonomousTaskCompletion,
    AutonomousTaskExecution,
    ManagerTurn,
)
from orchestration.system_events import SystemEvent

from .model_client import InferenceTiming

LOGGER = logging.getLogger(__name__)

MANAGER_PRIORITY = 0
HARDWARE_EVENT_PRIORITY = 10
DEFAULT_EVENT_PRIORITY = 100


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


@dataclass(slots=True)
class _TaskSliceResult:
    execution: AutonomousTaskExecution | None
    completion: AutonomousTaskCompletion | None


HumanTurnCallback = Callable[[ManagerTurn], None]
NotificationCallback = Callable[[ManagerTurn], None]
StatusCallback = Callable[[dict[str, object]], None]
ModelEventCallback = Callable[[str, str, str, InferenceTiming], None]


class CoreScheduler:
    """Headless priority scheduler owned by the long-lived cat-agent core."""

    def __init__(
        self,
        bundle,
        *,
        on_human_turn: HumanTurnCallback | None = None,
        on_notification: NotificationCallback | None = None,
        on_status: StatusCallback | None = None,
        on_model_event: ModelEventCallback | None = None,
        poll_interval: float = 0.02,
    ) -> None:
        self.bundle = bundle
        self.on_human_turn = on_human_turn
        self.on_notification = on_notification
        self.on_status = on_status
        self.on_model_event = on_model_event
        self.poll_interval = poll_interval

        self._queue_lock = threading.RLock()
        self._pending: list[_PriorityRequest] = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cat-agent")
        self._active_request: _PriorityRequest | None = None
        self._active_future: Future | None = None
        self._active_started: float | None = None
        self._background_request: _PriorityRequest | None = None
        self._background_execution: AutonomousTaskExecution | None = None
        self._active_is_task_slice = False
        self._last_request_seconds: float | None = None

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self.bundle.manager_client.set_event_handler(self._model_event)
        self._thread = threading.Thread(
            target=self._run,
            name="cat-agent-scheduler",
            daemon=True,
        )
        self._thread.start()
        self._emit_status()
        LOGGER.info("CORE scheduler started")

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        self.bundle.manager_client.set_event_handler(None)
        self._executor.shutdown(wait=True, cancel_futures=False)
        LOGGER.info("CORE scheduler stopped")

    def submit_user(self, text: str) -> None:
        user_text = text.strip()
        if not user_text:
            raise ValueError("user text must be non-empty")
        with self._queue_lock:
            self._pending.append(
                _PriorityRequest(
                    kind="user",
                    label="user",
                    payload=user_text,
                    queued_at=time.monotonic(),
                    priority=MANAGER_PRIORITY,
                )
            )
        LOGGER.info("CORE queued user request priority=%d text=%r", MANAGER_PRIORITY, user_text)
        self._emit_status()

    def enqueue_external_event(
        self,
        event: SystemEvent,
        *,
        priority: int = HARDWARE_EVENT_PRIORITY,
        coalesce: bool = False,
    ) -> None:
        self._validate_external_priority(priority)
        self._enqueue_event(
            event,
            priority=priority,
            coalesce_key=(self._event_label(event) if coalesce else None),
        )

    def status_snapshot(self) -> dict[str, object]:
        request = self._active_request
        timing = self.bundle.manager_client.inference_timing
        with self._queue_lock:
            pending = len(self._pending)
            background = self._background_request is not None
        return {
            "state": "BUSY" if self._active_future is not None else "IDLE",
            "label": request.label if request is not None else "",
            "priority": request.priority if request is not None else None,
            "request_started_monotonic": self._active_started,
            "last_request_seconds": self._last_request_seconds,
            "pending": pending,
            "background": background,
            "manager_resident_tokens": self.bundle.manager_client.resident_tokens,
            "agent_resident_tokens": self.bundle.agent_client.resident_tokens,
            "inference": self._timing_dict(timing),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_system_events()
                self._poll_future()
                self._start_next()
            except Exception:
                LOGGER.exception("CORE scheduler loop failed")
            self._stop.wait(self.poll_interval)

    def _poll_system_events(self) -> None:
        for event in self.bundle.system_runtime.poll_due(busy=False):
            coalesce_key = self._event_label(event) if event.source == "timer" else None
            self._enqueue_event(
                event,
                priority=DEFAULT_EVENT_PRIORITY,
                coalesce_key=coalesce_key,
            )

    def _enqueue_event(
        self,
        event: SystemEvent,
        *,
        priority: int,
        coalesce_key: str | None,
    ) -> None:
        label = self._event_label(event)
        with self._queue_lock:
            if coalesce_key is not None and self._has_coalesced_event(coalesce_key):
                LOGGER.info("CORE coalesced event label=%s", label)
                return
            self._pending.append(
                _PriorityRequest(
                    kind="system",
                    label=label,
                    payload=event,
                    queued_at=event.created_monotonic,
                    priority=priority,
                    coalesce_key=coalesce_key,
                )
            )
        LOGGER.info("CORE queued system event label=%s priority=%d", label, priority)
        self._emit_status()

    def _has_coalesced_event(self, key: str) -> bool:
        active = self._active_request
        future = self._active_future
        if (
            active is not None
            and (future is None or not future.done())
            and active.coalesce_key == key
        ):
            return True
        if self._background_request is not None and self._background_request.coalesce_key == key:
            return True
        return any(item.coalesce_key == key for item in self._pending)

    def _enqueue_manager_query_result(self, completion: AutonomousTaskCompletion) -> None:
        assert completion.query_task_id is not None
        request = _PriorityRequest(
            kind="manager",
            label=f"query-result:task:{completion.query_task_id}",
            payload=_ManagerQueryResult(
                task_id=completion.query_task_id,
                result=completion.query_result,
            ),
            queued_at=time.monotonic(),
            priority=MANAGER_PRIORITY,
        )
        with self._queue_lock:
            self._pending.append(request)
        LOGGER.info(
            "CORE queued manager query result task=%d priority=%d",
            completion.query_task_id,
            MANAGER_PRIORITY,
        )
        self._emit_status()

    def _start_next(self) -> None:
        if self._active_future is not None:
            return

        pending = self._peek_next_request()
        background = self._background_request
        execution = self._background_execution

        if background is not None and execution is not None:
            if pending is None or pending.priority >= background.priority:
                self._start_background_slice(background, execution)
                return

        request = self._take_next_request()
        if request is None:
            return

        if request.kind == "system":
            assert isinstance(request.payload, SystemEvent)
            event = request.payload
            if event.source == "timer" and not self.bundle.system_runtime.timer_enabled(event.name):
                LOGGER.info("CORE dropped stale timer event name=%s", event.name)
                return
            if event.task_id is not None and background is None:
                self._start_first_task_slice(request, event)
                return

        self._start_regular_request(request)

    def _start_regular_request(self, request: _PriorityRequest) -> None:
        self._active_request = request
        self._active_started = time.monotonic()
        self._active_is_task_slice = False
        LOGGER.info(
            "CORE request start kind=%s label=%s priority=%d queued_for=%.3fs",
            request.kind,
            request.label,
            request.priority,
            self._active_started - request.queued_at,
        )

        if request.kind == "user":
            assert isinstance(request.payload, str)
            self._active_future = self._executor.submit(
                self.bundle.runtime.user_message,
                request.payload,
            )
        elif request.kind == "manager":
            assert isinstance(request.payload, _ManagerQueryResult)
            self._active_future = self._executor.submit(
                self.bundle.runtime.autonomous_query_result,
                request.payload.task_id,
                request.payload.result,
            )
        else:
            assert isinstance(request.payload, SystemEvent)
            self._active_future = self._executor.submit(
                self.bundle.runtime.system_event,
                request.payload,
            )
        self._emit_status()

    def _start_first_task_slice(self, request: _PriorityRequest, event: SystemEvent) -> None:
        self._active_request = request
        self._active_started = time.monotonic()
        self._active_is_task_slice = True
        LOGGER.info(
            "CORE task slice start label=%s priority=%d queued_for=%.3fs",
            request.label,
            request.priority,
            self._active_started - request.queued_at,
        )
        self._active_future = self._executor.submit(self._run_first_task_slice, event)
        self._emit_status()

    def _start_background_slice(
        self,
        request: _PriorityRequest,
        execution: AutonomousTaskExecution,
    ) -> None:
        self._background_request = None
        self._background_execution = None
        self._active_request = request
        self._active_started = time.monotonic()
        self._active_is_task_slice = True
        LOGGER.info(
            "CORE task slice resume label=%s priority=%d age=%.3fs",
            request.label,
            request.priority,
            self._active_started - request.queued_at,
        )
        self._active_future = self._executor.submit(self._run_next_task_slice, execution)
        self._emit_status()

    def _run_first_task_slice(self, event: SystemEvent) -> _TaskSliceResult:
        started = self.bundle.runtime.begin_autonomous_task(event)
        if isinstance(started, AutonomousTaskCompletion):
            return _TaskSliceResult(None, started)
        completion = self.bundle.runtime.step_autonomous_task(started)
        if completion is None:
            return _TaskSliceResult(started, None)
        return _TaskSliceResult(None, completion)

    def _run_next_task_slice(self, execution: AutonomousTaskExecution) -> _TaskSliceResult:
        completion = self.bundle.runtime.step_autonomous_task(execution)
        if completion is None:
            return _TaskSliceResult(execution, None)
        return _TaskSliceResult(None, completion)

    def _poll_future(self) -> None:
        future = self._active_future
        if future is None or not future.done():
            return
        if self._active_is_task_slice:
            self._finish_task_slice(future)
        else:
            self._finish_regular_request(future)

    def _finish_regular_request(self, future: Future) -> None:
        request = self._active_request
        started = self._active_started
        try:
            turn = future.result()
        except Exception as exc:
            LOGGER.exception("CORE request failed")
            turn = ManagerTurn("error", str(exc))

        if request is not None and request.kind == "user":
            self._emit_human_turn(turn)
        elif turn.kind == "reply" and turn.text:
            self._emit_notification(turn)
        elif turn.kind == "error":
            LOGGER.error(
                "CORE background request failed label=%s error=%s",
                request.label if request is not None else "?",
                turn.text,
            )

        LOGGER.info(
            "CORE request complete kind=%s label=%s turn=%s text=%r",
            request.kind if request is not None else "?",
            request.label if request is not None else "?",
            turn.kind,
            turn.text,
        )
        self._finish_active(started)

    def _finish_task_slice(self, future: Future) -> None:
        request = self._active_request
        started = self._active_started
        try:
            result = future.result()
        except Exception as exc:
            LOGGER.exception("CORE task slice failed")
            result = _TaskSliceResult(
                None,
                AutonomousTaskCompletion(turn=ManagerTurn("error", str(exc))),
            )

        if result.execution is not None:
            assert request is not None
            self._background_request = request
            self._background_execution = result.execution
            LOGGER.info(
                "CORE task slice paused label=%s agent=%s",
                request.label,
                result.execution.worker.agent_id,
            )
        else:
            completion = result.completion or AutonomousTaskCompletion(
                turn=ManagerTurn("silent", "")
            )
            if completion.query_task_id is not None:
                self._enqueue_manager_query_result(completion)
                LOGGER.info(
                    "CORE task activation agent phase complete label=%s query-result queued",
                    request.label if request is not None else "?",
                )
            else:
                turn = completion.turn or ManagerTurn("silent", "")
                if turn.kind == "reply" and turn.text:
                    self._emit_notification(turn)
                elif turn.kind == "error":
                    LOGGER.error(
                        "CORE task activation failed label=%s error=%s",
                        request.label if request is not None else "?",
                        turn.text,
                    )
                LOGGER.info(
                    "CORE task activation complete label=%s turn=%s text=%r",
                    request.label if request is not None else "?",
                    turn.kind,
                    turn.text,
                )
        self._finish_active(started)

    def _finish_active(self, started: float | None) -> None:
        if started is not None:
            self._last_request_seconds = time.monotonic() - started
        self._active_future = None
        self._active_request = None
        self._active_started = None
        self._active_is_task_slice = False
        self._emit_status()

    def _peek_next_request(self) -> _PriorityRequest | None:
        with self._queue_lock:
            if not self._pending:
                return None
            return self._pending[self._best_pending_index()]

    def _take_next_request(self) -> _PriorityRequest | None:
        with self._queue_lock:
            if not self._pending:
                return None
            index = self._best_pending_index()
            return self._pending.pop(index)

    def _best_pending_index(self) -> int:
        return min(
            range(len(self._pending)),
            key=lambda index: (
                self._pending[index].priority,
                self._pending[index].queued_at,
                index,
            ),
        )

    def _model_event(self, label: str, event: str, payload: str) -> None:
        callback = self.on_model_event
        if callback is not None:
            try:
                callback(label, event, payload, self.bundle.manager_client.inference_timing)
            except Exception:
                LOGGER.exception("CORE model event callback failed")
        self._emit_status()

    def _emit_human_turn(self, turn: ManagerTurn) -> None:
        callback = self.on_human_turn
        if callback is not None:
            try:
                callback(turn)
            except Exception:
                LOGGER.exception("CORE human turn callback failed")

    def _emit_notification(self, turn: ManagerTurn) -> None:
        callback = self.on_notification
        if callback is not None:
            try:
                callback(turn)
            except Exception:
                LOGGER.exception("CORE notification callback failed")

    def _emit_status(self) -> None:
        callback = self.on_status
        if callback is not None:
            try:
                callback(self.status_snapshot())
            except Exception:
                LOGGER.debug("CORE status callback failed", exc_info=True)

    @staticmethod
    def _event_label(event: SystemEvent) -> str:
        return f"{event.source}:{event.name}"

    @staticmethod
    def _validate_external_priority(priority: int) -> None:
        if priority <= MANAGER_PRIORITY:
            raise ValueError(
                f"external event priority must be > {MANAGER_PRIORITY}; got {priority}"
            )

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
