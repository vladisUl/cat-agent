from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import shlex
import threading
import time

from .tasks import TaskRecord, TaskStore, TaskStoreError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SystemEvent:
    source: str
    name: str
    task: str
    created_monotonic: float
    task_id: int | None = None

    def manager_text(self) -> str:
        if self.task_id is not None:
            return (
                f"[SYSTEM_EVENT]\n"
                f"source: {self.source}\n"
                f"task_id: {self.task_id}\n"
                f"[/SYSTEM_EVENT]"
            )
        return (
            f"[SYSTEM_EVENT]\n"
            f"source: {self.source}\n"
            f"name: {self.name}\n"
            f"task:\n{self.task.strip()}\n"
            f"[/SYSTEM_EVENT]"
        )


@dataclass(frozen=True, slots=True)
class TaskActivation:
    source: str
    name: str
    task: TaskRecord
    created_monotonic: float


TaskHandler = Callable[[TaskActivation], str | None]


@dataclass(slots=True)
class TimerSpec:
    name: str
    period_seconds: float
    task: str
    enabled: bool
    next_fire_monotonic: float | None
    sequence: int = 0
    fired: int = 0
    skipped: int = 0


@dataclass(slots=True)
class TaskTimerSpec:
    task_id: int
    period_seconds: float
    enabled: bool
    next_fire_monotonic: float | None
    sequence: int = 0
    fired: int = 0
    skipped: int = 0


class SystemRuntime:
    """Internal system role: owns tasks, event sources and scheduling state."""

    def __init__(self, task_store: TaskStore | None = None) -> None:
        self._lock = threading.RLock()
        self._timers: dict[str, TimerSpec] = {}
        self._task_timers: dict[int, TaskTimerSpec] = {}
        self._task_store = task_store
        self._task_handler: TaskHandler | None = None
        self._source_state = {
            "timer": "ready",
            "gpio": "stub",
            "mqtt": "stub",
        }
        self._restore_task_timers()

    @property
    def task_store(self) -> TaskStore | None:
        return self._task_store

    def set_task_handler(self, handler: TaskHandler | None) -> None:
        with self._lock:
            self._task_handler = handler

    def arm_task_timers(self, now: float | None = None) -> None:
        """Start countdowns for enabled TASK timers after the runtime is fully ready."""
        current = time.monotonic() if now is None else now
        with self._lock:
            for timer in self._task_timers.values():
                timer.next_fire_monotonic = (
                    current + timer.period_seconds if timer.enabled else None
                )
                LOGGER.info(
                    "SYSTEM task timer armed id=%d period=%.3fs enabled=%s next=%s",
                    timer.task_id,
                    timer.period_seconds,
                    timer.enabled,
                    (
                        f"{timer.next_fire_monotonic:.3f}"
                        if timer.next_fire_monotonic is not None
                        else "none"
                    ),
                )

    def create_task(
        self,
        description: str,
        text: str,
        *,
        skills: tuple[str, ...] = (),
    ) -> TaskRecord:
        store = self._require_task_store()
        task = store.create(description, text, skills=skills)
        LOGGER.info("SYSTEM task created id=%d description=%r", task.task_id, task.description)
        return task

    def create_periodic_task(
        self,
        description: str,
        text: str,
        skills: tuple[str, ...],
        period_seconds: float,
        *,
        method: str = "task",
    ) -> TaskRecord:
        if not skills:
            raise TaskStoreError("periodic task requires at least one skill")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be > 0")

        store = self._require_task_store()
        task = store.create(
            description,
            text,
            method=method,
            skills=skills,
            timer_period_seconds=float(period_seconds),
            enabled=True,
        )
        now = time.monotonic()
        with self._lock:
            self._task_timers[task.task_id] = TaskTimerSpec(
                task_id=task.task_id,
                period_seconds=float(period_seconds),
                enabled=True,
                next_fire_monotonic=now + float(period_seconds),
            )
        LOGGER.info(
            "SYSTEM periodic task created id=%d method=%s period=%.3fs skills=%s description=%r",
            task.task_id,
            task.method,
            period_seconds,
            ",".join(skills),
            task.description,
        )
        return task

    def delete_task(self, task_id: int) -> bool:
        store = self._require_task_store()
        with self._lock:
            self._task_timers.pop(task_id, None)
        deleted = store.delete(task_id)
        LOGGER.info("SYSTEM task delete id=%d deleted=%s", task_id, deleted)
        return deleted

    def start_task(self, task_id: int) -> TaskRecord:
        store = self._require_task_store()
        task = store.require(task_id)
        if task.timer_period_seconds is None:
            raise TaskStoreError(f"task {task_id} has no timer")
        task = store.set_enabled(task_id, True)
        now = time.monotonic()
        with self._lock:
            timer = self._task_timers.get(task_id)
            if timer is None:
                timer = TaskTimerSpec(
                    task_id=task_id,
                    period_seconds=task.timer_period_seconds,
                    enabled=True,
                    next_fire_monotonic=now + task.timer_period_seconds,
                )
                self._task_timers[task_id] = timer
            else:
                timer.period_seconds = task.timer_period_seconds
                timer.enabled = True
                timer.next_fire_monotonic = now + task.timer_period_seconds
        LOGGER.info("SYSTEM task timer started id=%d", task_id)
        return task

    def stop_task(self, task_id: int) -> TaskRecord:
        store = self._require_task_store()
        task = store.require(task_id)
        if task.timer_period_seconds is None:
            raise TaskStoreError(f"task {task_id} has no timer")
        task = store.set_enabled(task_id, False)
        with self._lock:
            timer = self._task_timers.get(task_id)
            if timer is not None:
                timer.enabled = False
                timer.next_fire_monotonic = None
        LOGGER.info("SYSTEM task timer stopped id=%d", task_id)
        return task

    def set_task_period(self, task_id: int, period_seconds: float) -> TaskRecord:
        store = self._require_task_store()
        task = store.set_timer_period(task_id, period_seconds)
        now = time.monotonic()
        with self._lock:
            timer = self._task_timers.get(task_id)
            if timer is None:
                timer = TaskTimerSpec(
                    task_id=task_id,
                    period_seconds=period_seconds,
                    enabled=task.enabled,
                    next_fire_monotonic=(now + period_seconds if task.enabled else None),
                )
                self._task_timers[task_id] = timer
            else:
                timer.period_seconds = period_seconds
                timer.enabled = task.enabled
                timer.next_fire_monotonic = (
                    now + period_seconds if task.enabled else None
                )
        LOGGER.info("SYSTEM task timer period id=%d period=%.3fs", task_id, period_seconds)
        return task

    def task_snapshot(self) -> tuple[TaskRecord, ...]:
        store = self._require_task_store()
        return store.list()

    def task_status_text(self) -> str:
        store = self._require_task_store()
        return store.status_text()

    def activate_task(
        self,
        task_id: int,
        *,
        source: str,
        name: str = "",
        now: float | None = None,
    ) -> str | None:
        """Resolve TASK from persistent storage, run it, and return QUERY value if any."""
        task = self._require_task_store().require(task_id)
        activation = TaskActivation(
            source=source.strip() or "system",
            name=name.strip(),
            task=task,
            created_monotonic=time.monotonic() if now is None else now,
        )
        with self._lock:
            handler = self._task_handler

        LOGGER.info(
            "SYSTEM task activation id=%d method=%s source=%s name=%s description=%r",
            task.task_id,
            task.method,
            activation.source,
            activation.name,
            task.description,
        )
        if handler is None:
            LOGGER.info("SYSTEM task activation id=%d has no handler yet", task.task_id)
            return None
        return handler(activation)

    def capabilities_text(self) -> str:
        return "\n".join(
            [
                "timer: ready",
                "gpio: stub",
                "mqtt: stub",
            ]
        )

    def execute(self, command: str, body: str) -> str:
        """Legacy SYSTEM command path retained until manager prompt is switched."""
        LOGGER.info(
            "SYSTEM command=%r body=%r",
            command,
            " ".join(body.strip().split())[:1000],
        )
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            result = f"SYSTEM_ERROR\ninvalid command syntax: {exc}"
            LOGGER.warning("SYSTEM result=%r", result)
            return result

        if not argv:
            result = "SYSTEM_ERROR\nempty system command"
            LOGGER.warning("SYSTEM result=%r", result)
            return result
        if argv[0].upper() != "TIMER":
            source = argv[0].lower()
            state = self._source_state.get(source)
            if state == "stub":
                result = (
                    f"SYSTEM_ERROR\n{source} event source is reserved "
                    "but not implemented yet"
                )
            else:
                result = f"SYSTEM_ERROR\nunknown system source: {argv[0]}"
            LOGGER.warning("SYSTEM result=%r", result)
            return result

        result = self._execute_timer(argv[1:], body)
        if result.startswith("SYSTEM_OK"):
            LOGGER.info("SYSTEM result=%r", result)
        else:
            LOGGER.warning("SYSTEM result=%r", result)
        return result

    def _execute_timer(self, argv: list[str], body: str) -> str:
        if not argv:
            return "SYSTEM_ERROR\nTIMER requires an operation"

        op = argv[0].upper()

        if op == "SET":
            if len(argv) != 3:
                return "SYSTEM_ERROR\nusage: TIMER SET <name> <period_seconds>"
            name = argv[1]
            period = self._positive_seconds(argv[2])
            if period is None:
                return "SYSTEM_ERROR\nperiod_seconds must be > 0"
            task = body.strip()
            if not task:
                return "SYSTEM_ERROR\nTIMER SET requires the event task on following lines"
            now = time.monotonic()
            with self._lock:
                self._timers[name] = TimerSpec(
                    name=name,
                    period_seconds=period,
                    task=task,
                    enabled=True,
                    next_fire_monotonic=now + period,
                )
            LOGGER.info(
                "SYSTEM timer set name=%s period=%.3fs next=%.3f task=%r",
                name,
                period,
                now + period,
                " ".join(task.split())[:1000],
            )
            return f"SYSTEM_OK\ntimer {name} set to {period:g}s and started"

        if op == "START":
            if len(argv) != 2:
                return "SYSTEM_ERROR\nusage: TIMER START <name>"
            name = argv[1]
            now = time.monotonic()
            with self._lock:
                timer = self._timers.get(name)
                if timer is None:
                    return f"SYSTEM_ERROR\nunknown timer: {name}"
                timer.enabled = True
                timer.next_fire_monotonic = now + timer.period_seconds
            LOGGER.info("SYSTEM timer started name=%s", name)
            return f"SYSTEM_OK\ntimer {name} started"

        if op == "STOP":
            if len(argv) != 2:
                return "SYSTEM_ERROR\nusage: TIMER STOP <name>"
            name = argv[1]
            with self._lock:
                timer = self._timers.get(name)
                if timer is None:
                    return f"SYSTEM_ERROR\nunknown timer: {name}"
                timer.enabled = False
                timer.next_fire_monotonic = None
            LOGGER.info("SYSTEM timer stopped name=%s", name)
            return f"SYSTEM_OK\ntimer {name} stopped"

        if op == "PERIOD":
            if len(argv) != 3:
                return "SYSTEM_ERROR\nusage: TIMER PERIOD <name> <period_seconds>"
            name = argv[1]
            period = self._positive_seconds(argv[2])
            if period is None:
                return "SYSTEM_ERROR\nperiod_seconds must be > 0"
            now = time.monotonic()
            with self._lock:
                timer = self._timers.get(name)
                if timer is None:
                    return f"SYSTEM_ERROR\nunknown timer: {name}"
                timer.period_seconds = period
                if timer.enabled:
                    timer.next_fire_monotonic = now + period
            LOGGER.info("SYSTEM timer period name=%s period=%.3fs", name, period)
            return f"SYSTEM_OK\ntimer {name} period changed to {period:g}s"

        if op == "DELETE":
            if len(argv) != 2:
                return "SYSTEM_ERROR\nusage: TIMER DELETE <name>"
            name = argv[1]
            with self._lock:
                if self._timers.pop(name, None) is None:
                    return f"SYSTEM_ERROR\nunknown timer: {name}"
            LOGGER.info("SYSTEM timer deleted name=%s", name)
            return f"SYSTEM_OK\ntimer {name} deleted"

        if op == "LIST":
            if len(argv) != 1:
                return "SYSTEM_ERROR\nusage: TIMER LIST"
            return f"SYSTEM_OK\n{self.timer_status_text()}"

        return f"SYSTEM_ERROR\nunknown TIMER operation: {op}"

    def poll_due(
        self,
        now: float | None = None,
        *,
        busy: bool = False,
    ) -> tuple[SystemEvent, ...]:
        current = time.monotonic() if now is None else now
        events: list[SystemEvent] = []
        with self._lock:
            for timer in self._timers.values():
                if (
                    not timer.enabled
                    or timer.next_fire_monotonic is None
                    or current < timer.next_fire_monotonic
                ):
                    continue

                timer.sequence += 1
                if busy:
                    timer.skipped += 1
                    LOGGER.info(
                        "SYSTEM timer tick skipped name=%s sequence=%d reason=runtime_busy",
                        timer.name,
                        timer.sequence,
                    )
                else:
                    timer.fired += 1
                    LOGGER.info(
                        "SYSTEM timer tick emitted name=%s sequence=%d",
                        timer.name,
                        timer.sequence,
                    )
                    events.append(
                        SystemEvent(
                            source="timer",
                            name=timer.name,
                            task=timer.task,
                            created_monotonic=current,
                        )
                    )

                next_fire = timer.next_fire_monotonic
                while next_fire <= current:
                    next_fire += timer.period_seconds
                timer.next_fire_monotonic = next_fire

            for timer in self._task_timers.values():
                if (
                    not timer.enabled
                    or timer.next_fire_monotonic is None
                    or current < timer.next_fire_monotonic
                ):
                    continue

                timer.sequence += 1
                if busy:
                    timer.skipped += 1
                    LOGGER.info(
                        "SYSTEM task timer tick skipped task=%d sequence=%d reason=runtime_busy",
                        timer.task_id,
                        timer.sequence,
                    )
                else:
                    timer.fired += 1
                    LOGGER.info(
                        "SYSTEM task timer tick emitted task=%d sequence=%d",
                        timer.task_id,
                        timer.sequence,
                    )
                    events.append(
                        SystemEvent(
                            source="timer",
                            name=f"task:{timer.task_id}",
                            task="",
                            created_monotonic=current,
                            task_id=timer.task_id,
                        )
                    )

                next_fire = timer.next_fire_monotonic
                while next_fire <= current:
                    next_fire += timer.period_seconds
                timer.next_fire_monotonic = next_fire

        return tuple(events)

    def timer_enabled(self, name: str) -> bool:
        if name.startswith("task:"):
            try:
                task_id = int(name.split(":", 1)[1])
            except ValueError:
                return False
            with self._lock:
                timer = self._task_timers.get(task_id)
                return bool(timer is not None and timer.enabled)
        with self._lock:
            timer = self._timers.get(name)
            return bool(timer is not None and timer.enabled)

    def timer_snapshot(self) -> tuple[TimerSpec, ...]:
        with self._lock:
            return tuple(
                TimerSpec(
                    name=item.name,
                    period_seconds=item.period_seconds,
                    task=item.task,
                    enabled=item.enabled,
                    next_fire_monotonic=item.next_fire_monotonic,
                    sequence=item.sequence,
                    fired=item.fired,
                    skipped=item.skipped,
                )
                for item in self._timers.values()
            )

    def task_timer_snapshot(self) -> tuple[TaskTimerSpec, ...]:
        with self._lock:
            return tuple(
                TaskTimerSpec(
                    task_id=item.task_id,
                    period_seconds=item.period_seconds,
                    enabled=item.enabled,
                    next_fire_monotonic=item.next_fire_monotonic,
                    sequence=item.sequence,
                    fired=item.fired,
                    skipped=item.skipped,
                )
                for item in self._task_timers.values()
            )

    def timer_status_text(self) -> str:
        timers = self.timer_snapshot()
        if not timers:
            return "no timers"
        lines = []
        now = time.monotonic()
        for timer in sorted(timers, key=lambda item: item.name):
            if timer.enabled and timer.next_fire_monotonic is not None:
                remaining = max(0.0, timer.next_fire_monotonic - now)
                state = f"running next={remaining:.1f}s"
            else:
                state = "stopped"
            lines.append(
                f"{timer.name}: period={timer.period_seconds:g}s {state} "
                f"fired={timer.fired} skipped={timer.skipped} task={timer.task!r}"
            )
        return "\n".join(lines)

    def _restore_task_timers(self) -> None:
        if self._task_store is None:
            return
        for task in self._task_store.list():
            period = task.timer_period_seconds
            if period is None:
                continue
            self._task_timers[task.task_id] = TaskTimerSpec(
                task_id=task.task_id,
                period_seconds=period,
                enabled=task.enabled,
                next_fire_monotonic=None,
            )
            LOGGER.info(
                "SYSTEM restored task timer id=%d period=%.3fs enabled=%s state=unarmed",
                task.task_id,
                period,
                task.enabled,
            )

    def _require_task_store(self) -> TaskStore:
        if self._task_store is None:
            raise TaskStoreError("task store is not configured")
        return self._task_store

    @staticmethod
    def _positive_seconds(raw: str) -> float | None:
        try:
            value = float(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value
