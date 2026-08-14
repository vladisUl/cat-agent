from __future__ import annotations

from dataclasses import dataclass
import shlex
import threading
import time


@dataclass(frozen=True, slots=True)
class SystemEvent:
    source: str
    name: str
    task: str
    created_monotonic: float

    def manager_text(self) -> str:
        return (
            f"[SYSTEM_EVENT]\n"
            f"source: {self.source}\n"
            f"name: {self.name}\n"
            f"task:\n{self.task.strip()}\n"
            f"[/SYSTEM_EVENT]"
        )


@dataclass(slots=True)
class TimerSpec:
    name: str
    period_seconds: float
    task: str
    enabled: bool
    next_fire_monotonic: float | None


class SystemRuntime:
    """Internal system role: owns event sources and low-level scheduling state."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._timers: dict[str, TimerSpec] = {}
        self._source_state = {
            "timer": "ready",
            "gpio": "stub",
            "mqtt": "stub",
        }

    def capabilities_text(self) -> str:
        return "\n".join(
            [
                "timer: ready",
                "gpio: stub",
                "mqtt: stub",
            ]
        )

    def execute(self, command: str, body: str) -> str:
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            return f"SYSTEM_ERROR\ninvalid command syntax: {exc}"

        if not argv:
            return "SYSTEM_ERROR\nempty system command"
        if argv[0].upper() != "TIMER":
            source = argv[0].lower()
            state = self._source_state.get(source)
            if state == "stub":
                return f"SYSTEM_ERROR\n{source} event source is reserved but not implemented yet"
            return f"SYSTEM_ERROR\nunknown system source: {argv[0]}"

        return self._execute_timer(argv[1:], body)

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
            return (
                f"SYSTEM_OK\n"
                f"timer {name} set to {period:g}s and started"
            )

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
            return f"SYSTEM_OK\ntimer {name} period changed to {period:g}s"

        if op == "DELETE":
            if len(argv) != 2:
                return "SYSTEM_ERROR\nusage: TIMER DELETE <name>"
            name = argv[1]
            with self._lock:
                if self._timers.pop(name, None) is None:
                    return f"SYSTEM_ERROR\nunknown timer: {name}"
            return f"SYSTEM_OK\ntimer {name} deleted"

        if op == "LIST":
            if len(argv) != 1:
                return "SYSTEM_ERROR\nusage: TIMER LIST"
            return f"SYSTEM_OK\n{self.timer_status_text()}"

        return f"SYSTEM_ERROR\nunknown TIMER operation: {op}"

    def poll_due(self, now: float | None = None) -> tuple[SystemEvent, ...]:
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

                events.append(
                    SystemEvent(
                        source="timer",
                        name=timer.name,
                        task=timer.task,
                        created_monotonic=current,
                    )
                )

                # Keep periodic cadence stable while avoiding a catch-up storm after
                # a long model call: advance to the first future deadline.
                next_fire = timer.next_fire_monotonic
                while next_fire <= current:
                    next_fire += timer.period_seconds
                timer.next_fire_monotonic = next_fire

        return tuple(events)

    def timer_snapshot(self) -> tuple[TimerSpec, ...]:
        with self._lock:
            return tuple(
                TimerSpec(
                    name=item.name,
                    period_seconds=item.period_seconds,
                    task=item.task,
                    enabled=item.enabled,
                    next_fire_monotonic=item.next_fire_monotonic,
                )
                for item in self._timers.values()
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
                f"{timer.name}: period={timer.period_seconds:g}s {state} task={timer.task!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _positive_seconds(raw: str) -> float | None:
        try:
            value = float(raw)
        except ValueError:
            return None
        if value <= 0:
            return None
        return value
