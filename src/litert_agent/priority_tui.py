from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import threading
import time

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


class PriorityLiteRTTUI(LiteRTTUI):
    """LiteRT TUI with manager-first priority and coalesced background events."""

    def __init__(self, bundle) -> None:
        super().__init__(bundle)
        self._queue_lock = threading.RLock()

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
        # Timers must never disappear merely because the model is busy.
        # Coalescing below decides whether another activation of the same timer
        # is already pending/running.
        for event in self.bundle.system_runtime.poll_due(busy=False):
            self._enqueue_system_event(event)

    def _enqueue_system_event(self, event: SystemEvent) -> None:
        # Periodic timers are level-like: while one activation is pending or
        # running, later ticks of that same timer do not create a catch-up pile.
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

        while True:
            request = self._take_next_request()
            if request is None:
                return

            if request.kind == "system":
                assert isinstance(request.payload, SystemEvent)
                event = request.payload
                if (
                    event.source == "timer"
                    and not self.bundle.system_runtime.timer_enabled(event.name)
                ):
                    LOGGER.info(
                        "TUI dropped stale timer event name=%s because timer is stopped",
                        event.name,
                    )
                    continue
            break

        self._active_request = request
        self._active_started = time.monotonic()
        self._status = f"BUSY {request.label}"
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

    def _take_next_request(self) -> _PriorityRequest | None:
        with self._queue_lock:
            if not self._pending:
                return None

            best_index = min(
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
            request = self._pending[best_index]
            del self._pending[best_index]
            return request  # type: ignore[return-value]

    @staticmethod
    def _event_label(event: SystemEvent) -> str:
        return f"{event.source}:{event.name}"

    @staticmethod
    def _validate_external_priority(priority: int) -> None:
        if priority <= MANAGER_PRIORITY:
            raise ValueError(
                f"external event priority must be > {MANAGER_PRIORITY}; got {priority}"
            )
