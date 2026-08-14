from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import curses
import logging
from pathlib import Path
import sys
import textwrap
import time

from cat_agent.manager import ManagerTurn
from cat_agent.system_events import SystemEvent

from .runtime import LiteRTRuntimeBundle

LOGGER = logging.getLogger(__name__)
LOG_PATH = Path("/var/log/litertlm/cat-agent.log")
BRACKETED_PASTE_ON = "\x1b[?2004h"
BRACKETED_PASTE_OFF = "\x1b[?2004l"
BRACKETED_PASTE_START = "[200~"
BRACKETED_PASTE_END = "\x1b[201~"


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
        self._chat_started = time.monotonic()
        self._dialog: deque[tuple[str, str, int]] = deque(maxlen=1000)
        self._dialog_scroll_lines = 0
        self._dialog_page_lines = 10
        self._input = ""
        self._status = "IDLE"
        self._quit = False

    def run(self) -> None:
        self._set_bracketed_paste(True)
        try:
            curses.wrapper(self._main)
        finally:
            self._set_bracketed_paste(False)
            self._executor.shutdown(wait=True, cancel_futures=False)

    def _main(self, stdscr) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.timeout(100)
        self._append_dialog(
            "SYSTEM",
            "LiteRT ready. /quit exits. PageUp/PageDown scroll dialog.",
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

            if key == "\x1b" and self._try_bracketed_paste(stdscr):
                continue
            self._handle_key(key)

    def _runtime_busy(self) -> bool:
        return self._active_future is not None or bool(self._pending)

    def _append_dialog(self, speaker: str, text: str) -> None:
        elapsed = max(0, int(time.monotonic() - self._chat_started))
        self._dialog.append((speaker, text, elapsed))

    def _poll_system_events(self) -> None:
        busy = self._runtime_busy()
        for event in self.bundle.system_runtime.poll_due(busy=busy):
            self._append_dialog("SYSTEM", f"event {event.source}:{event.name}")
            self._dialog_scroll_lines = 0
            self._enqueue_system_event(event)

    def _enqueue_system_event(self, event: SystemEvent) -> None:
        label = f"{event.source}:{event.name}"
        for index, request in enumerate(self._pending):
            if request.kind == "system" and request.label == label:
                self._pending[index] = _Request(
                    kind="system",
                    label=label,
                    payload=event,
                    queued_at=time.monotonic(),
                )
                LOGGER.info("TUI coalesced pending system event label=%s", label)
                return
        self._pending.append(
            _Request(
                kind="system",
                label=label,
                payload=event,
                queued_at=time.monotonic(),
            )
        )
        LOGGER.info("TUI queued system event label=%s", label)

    def _start_next(self) -> None:
        if self._active_future is not None or not self._pending:
            return

        request = self._pending.popleft()
        if request.kind == "system":
            assert isinstance(request.payload, SystemEvent)
            event = request.payload
            if event.source == "timer" and not self.bundle.system_runtime.timer_enabled(event.name):
                LOGGER.info(
                    "TUI dropped stale timer event name=%s because timer is stopped",
                    event.name,
                )
                return

        self._active_request = request
        self._active_started = time.monotonic()
        self._status = f"BUSY {request.label}"
        LOGGER.info(
            "TUI request start kind=%s label=%s queued_for=%.3fs",
            request.kind,
            request.label,
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

    def _poll_future(self) -> None:
        future = self._active_future
        if future is None or not future.done():
            return

        request = self._active_request
        started = self._active_started
        try:
            turn = future.result()
        except Exception as exc:
            LOGGER.exception("TUI request failed")
            self._append_dialog("ERROR", str(exc))
        else:
            if request is not None and request.kind == "system":
                prefix = "SYSTEM/MANAGER"
            else:
                prefix = "MANAGER"
            if turn.kind == "wait":
                self._append_dialog(prefix, "WAIT")
            else:
                self._append_dialog(prefix, turn.text)
            LOGGER.info(
                "TUI request complete kind=%s label=%s turn=%s text=%r",
                request.kind if request is not None else "?",
                request.label if request is not None else "?",
                turn.kind,
                turn.text,
            )

        self._dialog_scroll_lines = 0
        if started is not None:
            self._last_request_seconds = time.monotonic() - started

        self._active_future = None
        self._active_request = None
        self._active_started = None
        self._status = "IDLE"

    def _handle_key(self, key) -> None:
        if key in ("\n", "\r", curses.KEY_ENTER):
            self._submit_input()
            return

        if key in (curses.KEY_BACKSPACE, "\b", "\x7f"):
            self._input = self._input[:-1]
            return

        if key == curses.KEY_PPAGE:
            self._dialog_scroll_lines += self._dialog_page_lines
            return

        if key == curses.KEY_NPAGE:
            self._dialog_scroll_lines = max(
                0,
                self._dialog_scroll_lines - self._dialog_page_lines,
            )
            return

        if key == curses.KEY_END:
            self._dialog_scroll_lines = 0
            return

        if key == "\x03":
            self._quit = True
            return

        if key == "\x16":
            self._status = "PASTE: use terminal paste shortcut"
            return

        if isinstance(key, str) and key.isprintable():
            self._input += key

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
        self._pending.append(
            _Request(
                kind="user",
                label="user",
                payload=text,
                queued_at=time.monotonic(),
            )
        )

    def _try_bracketed_paste(self, stdscr) -> bool:
        stdscr.timeout(30)
        prefix = ""
        try:
            for _ in range(len(BRACKETED_PASTE_START)):
                try:
                    item = stdscr.get_wch()
                except curses.error:
                    return False
                if not isinstance(item, str):
                    return False
                prefix += item
            if prefix != BRACKETED_PASTE_START:
                return False

            stdscr.timeout(1000)
            data = ""
            tail = ""
            while True:
                try:
                    item = stdscr.get_wch()
                except curses.error:
                    LOGGER.warning("TUI bracketed paste timed out")
                    break
                if not isinstance(item, str):
                    continue
                tail += item
                if tail.endswith(BRACKETED_PASTE_END):
                    data += tail[: -len(BRACKETED_PASTE_END)]
                    break
                if len(tail) > len(BRACKETED_PASTE_END):
                    data += tail[0]
                    tail = tail[1:]

            pasted = self._normalize_paste(data)
            if pasted:
                self._input += pasted
                self._status = f"PASTED {len(pasted)} chars"
                LOGGER.info("TUI pasted %d characters", len(pasted))
            return True
        finally:
            stdscr.timeout(100)

    @staticmethod
    def _normalize_paste(text: str) -> str:
        return " ".join(text.replace("\r", "\n").splitlines()).strip()

    @staticmethod
    def _set_bracketed_paste(enabled: bool) -> None:
        sequence = BRACKETED_PASTE_ON if enabled else BRACKETED_PASTE_OFF
        try:
            sys.stdout.write(sequence)
            sys.stdout.flush()
        except (AttributeError, OSError):
            pass

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

        input_lines, input_cursor = self._layout_input(width)
        max_input_lines = max(2, min(10, height // 3))
        visible_input_lines = input_lines[-max_input_lines:]
        hidden_input_lines = len(input_lines) - len(visible_input_lines)
        input_height = len(visible_input_lines) + 2
        body_height = height - input_height + 1

        left_width = max(55, int(width * 0.64))
        right_x = left_width - 1
        right_width = width - right_x

        left = stdscr.derwin(body_height, left_width, 0, 0)
        right = stdscr.derwin(body_height, right_width, 0, right_x)
        input_y = body_height - 1
        input_win = stdscr.derwin(input_height, width, input_y, 0)

        self._draw_box(left)
        self._draw_box(right)
        self._draw_box(input_win)

        self._title(left, " DIALOG ")
        self._title(right, " LiteRT-LM ")
        self._title(input_win, " INPUT ")

        self._draw_dialog(left)
        self._draw_info(right)
        self._draw_input(
            input_win,
            visible_input_lines,
            input_cursor,
            hidden_input_lines,
        )

        left.noutrefresh()
        right.noutrefresh()
        input_win.noutrefresh()
        curses.doupdate()

    def _draw_dialog(self, win) -> None:
        height, width = win.getmaxyx()
        usable_width = max(1, width - 4)
        lines: list[tuple[str, str, bool]] = []

        for speaker, text, elapsed in self._dialog:
            normalized = " ".join(text.strip().split())
            timed_text = f"{normalized} ({self._format_elapsed(elapsed)})"
            prefix = f"{speaker}> "
            wrapped = textwrap.wrap(
                timed_text,
                width=max(10, usable_width - len(prefix)),
                replace_whitespace=True,
                drop_whitespace=True,
            ) or [""]
            lines.append((prefix, wrapped[0], True))
            indent = " " * len(prefix)
            lines.extend((indent, item, False) for item in wrapped[1:])
            lines.append(("", "", False))

        page = max(1, height - 2)
        self._dialog_page_lines = max(5, page - 2)
        max_scroll = max(0, len(lines) - page)
        self._dialog_scroll_lines = min(self._dialog_scroll_lines, max_scroll)
        end = len(lines) - self._dialog_scroll_lines
        start = max(0, end - page)
        visible = lines[start:end]

        for row, (lead, text, first_line) in enumerate(visible, start=1):
            if first_line:
                self._safe_addstr(win, row, 2, lead, curses.A_BOLD, width - 4)
                self._safe_addstr(
                    win,
                    row,
                    2 + len(lead),
                    text,
                    0,
                    width - 4 - len(lead),
                )
            else:
                self._safe_addstr(win, row, 2, lead + text, 0, width - 4)

        if self._dialog_scroll_lines:
            marker = f" ↑ {self._dialog_scroll_lines} lines "
            self._safe_addstr(win, 0, max(2, width - len(marker) - 2), marker, curses.A_BOLD)

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
                put("  fired/skip", f"{timer.fired}/{timer.skipped}")

        if self._pending:
            put("queue", str(len(self._pending)))

        row += 1
        put("LOG", bold=True)
        put("file", str(LOG_PATH))
        try:
            put("size", self._format_bytes(LOG_PATH.stat().st_size))
        except OSError:
            put("size", "--")

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

    def _layout_input(self, width: int) -> tuple[list[str], tuple[int, int]]:
        inner_width = max(8, width - 4)
        prompt = "> "
        first_capacity = max(1, inner_width - len(prompt))
        text = self._input

        first = text[:first_capacity]
        lines = [prompt + first]
        consumed = len(first)
        while consumed < len(text):
            chunk = text[consumed : consumed + inner_width]
            lines.append(chunk)
            consumed += len(chunk)

        if text and len(text) == first_capacity:
            lines.append("")
        elif len(text) > first_capacity and (len(text) - first_capacity) % inner_width == 0:
            lines.append("")

        cursor_row = len(lines) - 1
        cursor_col = len(lines[-1])
        return lines, (cursor_row, cursor_col)

    def _draw_input(
        self,
        win,
        lines: list[str],
        cursor: tuple[int, int],
        hidden_lines: int,
    ) -> None:
        _height, width = win.getmaxyx()
        for row, line in enumerate(lines, start=1):
            display = line
            if row == 1 and hidden_lines:
                display = "… " + display
            self._safe_addstr(win, row, 2, display, curses.A_BOLD, width - 4)

        cursor_row, cursor_col = cursor
        visible_cursor_row = cursor_row - hidden_lines + 1
        if visible_cursor_row < 1:
            visible_cursor_row = 1
            cursor_col = 0
        cursor_x = min(width - 2, 2 + cursor_col)
        self._safe_addstr(
            win,
            visible_cursor_row,
            cursor_x,
            " ",
            curses.A_REVERSE | curses.A_BOLD,
            1,
        )

    @staticmethod
    def _seconds(value: float | None) -> str:
        return "--" if value is None else f"{value:.3f}s"

    @staticmethod
    def _format_elapsed(value: int) -> str:
        total = max(0, int(value))
        if total < 60:
            return f"{total}сек"
        minutes, seconds = divmod(total, 60)
        if minutes < 60:
            return f"{minutes}мин{seconds}сек"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}ч{minutes}мин{seconds}сек"

    @staticmethod
    def _format_bytes(value: int) -> str:
        if value < 1024:
            return f"{value} B"
        if value < 1024 * 1024:
            return f"{value / 1024:.1f} KiB"
        return f"{value / (1024 * 1024):.1f} MiB"

    @staticmethod
    def _draw_box(win) -> None:
        height, width = win.getmaxyx()
        if height < 2 or width < 2:
            return
        top = "┌" + "─" * max(0, width - 2) + "┐"
        bottom = "└" + "─" * max(0, width - 2) + "┘"
        LiteRTTUI._safe_addstr(win, 0, 0, top)
        for row in range(1, height - 1):
            LiteRTTUI._safe_addstr(win, row, 0, "│")
            LiteRTTUI._safe_addstr(win, row, width - 1, "│")
        LiteRTTUI._safe_addstr(win, height - 1, 0, bottom)

    @staticmethod
    def _title(win, title: str) -> None:
        try:
            win.addstr(0, 2, title, curses.A_BOLD)
        except curses.error:
            pass

    @staticmethod
    def _safe_addstr(
        win,
        y: int,
        x: int,
        text: str,
        attr: int = 0,
        limit: int | None = None,
    ) -> None:
        try:
            if limit is None:
                win.addstr(y, x, text, attr)
            else:
                win.addnstr(y, x, text, max(0, limit), attr)
        except curses.error:
            pass
