from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import curses
import textwrap
import time

from cat_agent.manager import ManagerTurn
from cat_agent.system_events import SystemEvent

from .runtime import LiteRTRuntimeBundle


@dataclass(slots=True)
class _Request:
    kind: str
    label: str
    payload: str | SystemEvent
    queued_at: float


class LiteRTTUI:
    def __init__(self, bundle: LiteRTRuntimeBundle) -> None:
        self.bundle = bundle
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cat-agent")
        self._pending: deque[_Request] = deque()
        self._active_request: _Request | None = None
        self._active_future: Future[ManagerTurn] | None = None
        self._active_started: float | None = None
        self._last_request_seconds: float | None = None
        self._dialog: deque[tuple[str, str]] = deque(maxlen=500)
        self._input = ""
        self._status = "IDLE"
        self._quit = False

    def run(self) -> None:
        try:
            curses.wrapper(self._main)
        finally:
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _main(self, stdscr) -> None:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.timeout(100)
        self._dialog.append(
            (
                "SYSTEM",
                "LiteRT ready. /quit exits. Internal timer events are active.",
            )
        )

        while not self._quit:
            self._poll_system_events()
            self._poll_future()
            self._start_next()
            self._draw(stdscr)

            try:
                key = stdscr.get_wch()
            except curses.error:
                continue
            self._handle_key(key)

    def _poll_system_events(self) -> None:
        for event in self.bundle.system_runtime.poll_due():
            self._dialog.append(
                ("SYSTEM", f"event {event.source}:{event.name}")
            )
            self._pending.append(
                _Request(
                    kind="system",
                    label=f"{event.source}:{event.name}",
                    payload=event,
                    queued_at=time.monotonic(),
                )
            )

    def _start_next(self) -> None:
        if self._active_future is not None or not self._pending:
            return

        request = self._pending.popleft()
        self._active_request = request
        self._active_started = time.monotonic()
        self._status = f"BUSY {request.label}"

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

    def _poll_future(self) -> None:
        future = self._active_future
        if future is None or not future.done():
            return

        request = self._active_request
        started = self._active_started
        try:
            turn = future.result()
        except Exception as exc:
            self._dialog.append(("ERROR", str(exc)))
        else:
            if request is not None and request.kind == "system":
                prefix = "SYSTEM/MANAGER"
            else:
                prefix = "MANAGER"
            if turn.kind == "wait":
                self._dialog.append((prefix, "WAIT"))
            else:
                self._dialog.append((prefix, turn.text))

        if started is not None:
            self._last_request_seconds = time.monotonic() - started

        self._active_future = None
        self._active_request = None
        self._active_started = None
        self._status = "IDLE"

    def _handle_key(self, key) -> None:
        if key in ("\n", "\r", curses.KEY_ENTER):
            text = self._input.strip()
            self._input = ""
            if not text:
                return
            if text in {"/quit", "/exit"}:
                self._quit = True
                return
            self._dialog.append(("YOU", text))
            self._pending.append(
                _Request(
                    kind="user",
                    label="user",
                    payload=text,
                    queued_at=time.monotonic(),
                )
            )
            return

        if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
            self._input = self._input[:-1]
            return

        if key == "\x03":
            self._quit = True
            return

        if isinstance(key, str) and key.isprintable():
            self._input += key

    def _draw(self, stdscr) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        if height < 18 or width < 90:
            self._safe_addstr(
                stdscr,
                0,
                0,
                f"Terminal too small: need at least 90x18, current {width}x{height}",
                curses.A_BOLD,
            )
            stdscr.refresh()
            return

        input_height = 3
        body_height = height - input_height
        left_width = max(55, int(width * 0.64))
        right_width = width - left_width

        left = stdscr.derwin(body_height, left_width, 0, 0)
        right = stdscr.derwin(body_height, right_width, 0, left_width)
        input_win = stdscr.derwin(input_height, width, body_height, 0)

        left.box()
        right.box()
        input_win.box()

        self._title(left, " DIALOG ")
        self._title(right, " LiteRT-LM ")
        self._title(input_win, " INPUT ")

        self._draw_dialog(left)
        self._draw_info(right)
        self._draw_input(input_win)

        try:
            input_win.move(1, min(width - 2, 2 + len(self._input)))
            curses.curs_set(1)
        except curses.error:
            pass

        left.noutrefresh()
        right.noutrefresh()
        input_win.noutrefresh()
        curses.doupdate()

    def _draw_dialog(self, win) -> None:
        height, width = win.getmaxyx()
        usable_width = max(1, width - 4)
        lines: list[str] = []

        for speaker, text in self._dialog:
            normalized = " ".join(text.strip().split())
            prefix = f"{speaker}> "
            wrapped = textwrap.wrap(
                normalized,
                width=max(10, usable_width - len(prefix)),
                replace_whitespace=True,
                drop_whitespace=True,
            ) or [""]
            lines.append(prefix + wrapped[0])
            indent = " " * len(prefix)
            lines.extend(indent + item for item in wrapped[1:])
            lines.append("")

        visible = lines[-max(1, height - 2) :]
        for row, line in enumerate(visible, start=1):
            attr = curses.A_BOLD if line.startswith(("YOU>", "MANAGER>", "SYSTEM>")) else 0
            self._safe_addstr(win, row, 2, line, attr, width - 4)

    def _draw_info(self, win) -> None:
        height, width = win.getmaxyx()
        row = 1

        def put(label: str, value: str = "", *, bold: bool = False) -> None:
            nonlocal row
            if row >= height - 1:
                return
            text = f"{label:<12} {value}" if value else label
            self._safe_addstr(
                win,
                row,
                2,
                text,
                curses.A_BOLD if bold else 0,
                width - 4,
            )
            row += 1

        put("STATE", self._status, bold=True)
        if self._active_started is not None:
            put("elapsed", f"{time.monotonic() - self._active_started:.1f}s")
        elif self._last_request_seconds is not None:
            put("last total", f"{self._last_request_seconds:.3f}s")

        row += 1
        put("MODEL", self.bundle.model_path.name, bold=True)
        put("backend", self.bundle.backend_name.upper())
        put("speculative", "ON" if self.bundle.speculative else "OFF")
        put(
            "engine M/A",
            f"{self.bundle.manager_engine_init_seconds:.3f}s / "
            f"{self.bundle.agent_engine_init_seconds:.3f}s",
        )

        row += 1
        put("WARM", bold=True)
        if self.bundle.manager_warm is not None:
            put(
                "manager",
                f"{self.bundle.manager_warm.token_count} tok "
                f"{self.bundle.manager_warm.elapsed_seconds:.3f}s",
            )
        if self.bundle.agent_warm is not None:
            put(
                "agent",
                f"{self.bundle.agent_warm.token_count} tok "
                f"{self.bundle.agent_warm.elapsed_seconds:.3f}s",
            )

        row += 1
        put("MANAGER", bold=True)
        row = self._draw_client_stats(win, row, self.bundle.manager_client, width, height)

        row += 1
        put("AGENT", bold=True)
        row = self._draw_client_stats(win, row, self.bundle.agent_client, width, height)

        row += 1
        put("EVENTS", bold=True)
        put("sources", "timer=ready gpio=stub mqtt=stub")
        timers = self.bundle.system_runtime.timer_snapshot()
        if not timers:
            put("timers", "none")
        else:
            now = time.monotonic()
            for timer in sorted(timers, key=lambda item: item.name):
                if row >= height - 1:
                    break
                if timer.enabled and timer.next_fire_monotonic is not None:
                    next_in = max(0.0, timer.next_fire_monotonic - now)
                    state = f"{timer.period_seconds:g}s next {next_in:.1f}s"
                else:
                    state = f"{timer.period_seconds:g}s stopped"
                put(timer.name[:12], state)

        if self._pending:
            put("queue", str(len(self._pending)))

    def _draw_client_stats(self, win, row: int, client, width: int, height: int) -> int:
        def put(text: str) -> None:
            nonlocal row
            if row >= height - 1:
                return
            self._safe_addstr(win, row, 2, text, 0, width - 4)
            row += 1

        put(f"resident     {client.resident_tokens}")
        response = client.last_response
        if response is None:
            put("last         --")
            return row

        cached = response.cached_tokens if response.cached_tokens is not None else 0
        new = (
            response.prompt_evaluated_tokens
            if response.prompt_evaluated_tokens is not None
            else 0
        )
        decode = response.completion_tokens if response.completion_tokens is not None else 0
        put(f"cached/new   {cached} / {new}")
        put(f"decode       {decode}")
        put(f"prefill      {self._seconds(response.prompt_seconds)}")
        put(f"generate     {self._seconds(response.generation_seconds)}")
        put(f"wall         {response.elapsed_seconds:.3f}s")
        return row

    def _draw_input(self, win) -> None:
        height, width = win.getmaxyx()
        del height
        prompt = "> "
        available = max(1, width - 4 - len(prompt))
        visible = self._input[-available:]
        self._safe_addstr(win, 1, 2, prompt + visible, curses.A_BOLD, width - 4)

    @staticmethod
    def _seconds(value: float | None) -> str:
        return "--" if value is None else f"{value:.3f}s"

    @staticmethod
    def _title(win, title: str) -> None:
        try:
            win.addstr(0, 2, title, curses.A_BOLD)
        except curses.error:
            pass

    @staticmethod
    def _safe_addstr(win, y: int, x: int, text: str, attr: int = 0, limit: int | None = None) -> None:
        try:
            if limit is None:
                win.addstr(y, x, text, attr)
            else:
                win.addnstr(y, x, text, max(0, limit), attr)
        except curses.error:
            pass
