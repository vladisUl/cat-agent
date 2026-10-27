"""Single-thread priority scheduler for LiteRT Conversations.

A running native send_message() is non-preemptible. Pending operations are
strictly prioritized. This module has no legacy orchestration imports.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import heapq
import logging
import threading
from typing import Callable

LOGGER = logging.getLogger(__name__)

PRIORITIES = {"control": -10, "human": 0, "hardware": 10, "background": 20}


class QueueFullError(RuntimeError):
    """A new operation was NOT admitted; never acknowledge it as accepted."""


@dataclass(order=True)
class _Operation:
    priority: int
    sequence: int
    kind: str = field(compare=False)
    callback: Callable[[], None] = field(compare=False)


class PriorityScheduler:
    def __init__(self, *, max_pending: int = 16):
        if max_pending < 1:
            raise ValueError("max_pending must be >= 1")
        self._cv = threading.Condition()
        self._queue: list[_Operation] = []
        self._max_pending = max_pending
        self._sequence = 0
        self._stopping = False
        self._active: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="native-model-scheduler", daemon=True,
        )
        self._thread.start()

    def submit(self, kind: str, callback: Callable[[], None], *, continuation: bool = False) -> None:
        if kind not in PRIORITIES:
            raise ValueError(f"Unknown model operation kind: {kind}")
        with self._cv:
            if self._stopping:
                raise RuntimeError("Scheduler is stopped")
            # Mandatory KV cleanup and already-admitted background continuations
            # cannot be dropped because new requests filled the queue.
            if kind != "control" and not continuation and len(self._queue) >= self._max_pending:
                raise QueueFullError("CORE model queue is full")
            self._sequence += 1
            heapq.heappush(self._queue, _Operation(
                PRIORITIES[kind], self._sequence, kind, callback,
            ))
            self._cv.notify()

    def snapshot(self) -> dict[str, object]:
        with self._cv:
            counts = Counter(item.kind for item in self._queue)
            return {
                "state": "BUSY" if self._active else "IDLE",
                "pending": len(self._queue),
                "label": self._active or "",
                "queued_by_priority": {
                    k: counts[k] for k in ("human", "hardware", "background", "control")
                },
            }

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._stopping or bool(self._queue))
                if self._stopping:
                    return
                item = heapq.heappop(self._queue)
                self._active = item.kind
            try:
                item.callback()
            except BaseException:
                LOGGER.exception("Scheduled native operation failed kind=%s", item.kind)
            finally:
                with self._cv:
                    self._active = None
                    self._cv.notify_all()

    def close(self) -> None:
        with self._cv:
            self._stopping = True
            self._queue.clear()  # Never replay pending side effects on shutdown.
            self._cv.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join()
