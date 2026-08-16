from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import threading
import time

from cat_agent.manager import AutonomousTaskExecution, ManagerTurn
from cat_agent.system_events import SystemEvent

from .tui import LiteRTTUI

LOGGER = logging.getLogger(__name__)

MANAGER_PRIORITY = 0
DEFAULT_EVENT_PRIORITY = 100


@dataclass(slots=True)
class _PriorityRequest:
    kind: str
    label: str
    payload: str | SystemEvent
    queued_at: float
    priority: int
    coalesce_key: str | None = None


@dataclass(slots=True)
class _TaskSliceResult:
    execution: AutonomousTaskExecution | None
    turn: ManagerTurn | None


class PriorityLiteRTTUI(LiteRTTUI):
    """Manager-first scheduler with resumable autonomous agent TT work."""

    def __init__(self, bundle) -> None:
        super().__init__(bundle)
        self._queue_lock = threading.RLock()
        self._background_request: _PriorityRequest | None = None
        self._background_execution: AutonomousTaskExecution | None = None
        self._active_is_task_slice = False

    def make_event_callback(
        self,
        *,
        priority: int = DEFAULT_EVENT_PRIORITY,
        coalesce: bool = False,
    ) -> Callable[[SystemEvent], None]:
        """Build a callback for a future asynchronous event source."""
        self._validate_external_priority(priority)

        def callback(event: SystemEvent) -> None:
            self.enqueue_external_event(
                event,
                priority=priority,
                coalesce=coalesce,
            )

        return callback

    def enqueue_external_event(
        self,
        event: SystemEvent,
        *,
        priority: int = DEFAULT_EVENT_PRIORITY,
        coalesce: bool = False,
    ) -> None:
        """Queue one external event without coupling the scheduler to its source."""
        self._validate_external_priority(priority)
        self._enqueue_event(
            event,
            priority=priority,
            coalesce_key=(self._event_label(event) if coalesce else None),
        )

    def _poll_system_events(self) -> None:
        # Timers must never disappear merely because model work is in progress.
        for event in self.bundle.system_runtime.poll_due(busy=False):
            self._enqueue_system_event(event)

    def _enqueue_system_event(self, event: SystemEvent) -> None:
        # Periodic timers keep at most one activation pending or running.
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
            if coalesce_key is not None:
                active = self._active_request
                future = self._active_future
                active_running = active is not None and (
                    future is None or not future.done()
                )
                if (
                    active_running
                    and getattr(active, "coalesce_key", None) == coalesce_key
                ):
                    LOGGER.info(
                        "TUI coalesced system event label=%s state=running",
                        label,
                    )
                    return

                if (
                    self._background_request is not None
                    and self._background_request.coalesce_key == coalesce_key
                ):
                    LOGGER.info(
                        "TUI coalesced system event label=%s state=tt-paused",
                        label,
                    )
                    return

                for request in self._pending:
                    if getattr(request, "coalesce_key", None) == coalesce_key:
                        LOGGER.info(
                            "TUI coalesced system event label=%s state=pending",
                            label,
                        )
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

        LOGGER.info(
            "TUI queued system event label=%s priority=%d",
            label,
            priority,
        )

    def _submit_input(self) -> None:
        text = self._input.strip()
        self._input = ""
        if not text:
            return
        if text in {"/quit", "/exit"}:
            self._quit = True
            return

        LOGGER.info("USER input=%r", text)
        self._append_dialog("YOU", text)
        self._dialog_scroll_lines = 0
        with self._queue_lock:
            self._pending.append(
                _PriorityRequest(
                    kind="user",
                    label="user",
                    payload=text,
                    queued_at=time.monotonic(),
                    priority=MANAGER_PRIORITY,
                )
            )

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
                LOGGER.info(
                    "TUI dropped stale timer event name=%s because timer is stopped",
                    event.name,
                )
                return

            # A saved TASK/QUERY is cooperative: one executor job equals one
            # agent TT. If another autonomous task ever preempts an already
            # suspended one, keep the nested event synchronous for now rather
            # than overwrite the suspended context.
            if event.task_id is not None and background is None:
                self._start_first_task_slice(request, event)
                return

        self._start_regular_request(request)

    def _start_regular_request(self, request: _PriorityRequest) -> None:
        self._active_request = request
        self._active_started = time.monotonic()
        self._status = f"BUSY {request.label}"
        self._active_is_task_slice = False
        LOGGER.info(
            "TUI request start kind=%s label=%s priority=%d queued_for=%.3fs",
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
        else:
            assert isinstance(request.payload, SystemEvent)
            self._active_future = self._executor.submit(
                self.bundle.runtime.system_event,
                request.payload,
            )

    def _start_first_task_slice(
        self,
        request: _PriorityRequest,
        event: SystemEvent,
    ) -> None:
        self._active_request = request
        self._active_started = time.monotonic()
        self._status = f"BUSY {request.label} TT"
        self._active_is_task_slice = True
        LOGGER.info(
            "TUI task slice start label=%s priority=%d queued_for=%.3fs",
            request.label,
            request.priority,
            self._active_started - request.queued_at,
        )
        self._active_future = self._executor.submit(
            self._run_first_task_slice,
            event,
        )

    def _start_background_slice(
        self,
        request: _PriorityRequest,
        execution: AutonomousTaskExecution,
    ) -> None:
        self._background_request = None
        self._background_execution = None
        self._active_request = request
        self._active_started = time.monotonic()
        self._status = f"BUSY {request.label} TT"
        self._active_is_task_slice = True
        LOGGER.info(
            "TUI task slice resume label=%s priority=%d age=%.3fs",
            request.label,
            request.priority,
            self._active_started - request.queued_at,
        )
        self._active_future = self._executor.submit(
            self._run_next_task_slice,
            execution,
        )

    def _run_first_task_slice(self, event: SystemEvent) -> _TaskSliceResult:
        started = self.bundle.runtime.begin_autonomous_task(event)
        if isinstance(started, ManagerTurn):
            return _TaskSliceResult(None, started)
        turn = self.bundle.runtime.step_autonomous_task(started)
        if turn is None:
            return _TaskSliceResult(started, None)
        return _TaskSliceResult(None, turn)

    def _run_next_task_slice(
        self,
        execution: AutonomousTaskExecution,
    ) -> _TaskSliceResult:
        turn = self.bundle.runtime.step_autonomous_task(execution)
        if turn is None:
            return _TaskSliceResult(execution, None)
        return _TaskSliceResult(None, turn)

    def _poll_future(self) -> None:
        if not self._active_is_task_slice:
            super()._poll_future()
            return

        future = self._active_future
        if future is None or not future.done():
            return

        request = self._active_request
        started = self._active_started
        dialog_updated = False
        try:
            result = future.result()
        except Exception as exc:
            LOGGER.exception("TUI task slice failed")
            turn = ManagerTurn("error", str(exc))
            result = _TaskSliceResult(None, turn)

        if result.execution is not None:
            assert request is not None
            assert isinstance(request, _PriorityRequest)
            self._background_request = request
            self._background_execution = result.execution
            LOGGER.info(
                "TUI task slice paused label=%s agent=%s",
                request.label,
                result.execution.worker.agent_id,
            )
        else:
            turn = result.turn or ManagerTurn("silent", "")
            if turn.kind == "reply" and turn.text:
                self._append_dialog("MANAGER", turn.text)
                dialog_updated = True
            LOGGER.info(
                "TUI task activation complete label=%s turn=%s text=%r",
                request.label if request is not None else "?",
                turn.kind,
                turn.text,
            )

        if dialog_updated:
            self._dialog_scroll_lines = 0
        if started is not None:
            self._last_request_seconds = time.monotonic() - started

        self._active_future = None
        self._active_request = None
        self._active_started = None
        self._active_is_task_slice = False
        self._status = "IDLE"

    def _peek_next_request(self) -> _PriorityRequest | None:
        with self._queue_lock:
            if not self._pending:
                return None
            index = self._best_pending_index()
            return self._pending[index]  # type: ignore[return-value]

    def _take_next_request(self) -> _PriorityRequest | None:
        with self._queue_lock:
            if not self._pending:
                return None
            index = self._best_pending_index()
            request = self._pending[index]
            del self._pending[index]
            return request  # type: ignore[return-value]

    def _best_pending_index(self) -> int:
        return min(
            range(len(self._pending)),
            key=lambda index: (
                getattr(
                    self._pending[index],
                    "priority",
                    DEFAULT_EVENT_PRIORITY,
                ),
                self._pending[index].queued_at,
                index,
            ),
        )

    @staticmethod
    def _event_label(event: SystemEvent) -> str:
        return f"{event.source}:{event.name}"

    @staticmethod
    def _validate_external_priority(priority: int) -> None:
        if priority <= MANAGER_PRIORITY:
            raise ValueError(
                f"external event priority must be > {MANAGER_PRIORITY}; got {priority}"
            )
