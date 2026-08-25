from __future__ import annotations

from collections import deque
import curses
import json
import os
from pathlib import Path
import queue
import socket
import sys
import textwrap
import threading
import time

from .core_server import DEFAULT_CORE_SOCKET

LOG_PATH = Path("/var/log/litertlm/cat-agent.log")
BRACKETED_PASTE_ON = "\x1b[?2004h"
BRACKETED_PASTE_OFF = "\x1b[?2004l"
BRACKETED_PASTE_START = "[200~"
BRACKETED_PASTE_END = "\x1b[201~"
COLOR_ACTIVE = 1
COLOR_STOPPED = 2
SNAPSHOT_INTERVAL = 0.2


class CoreClient:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.sock: socket.socket | None = None
        self.reader = None
        self.incoming: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
        self._send_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def connect(self) -> dict[str, object]:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(self.path))
        self.sock = sock
        self.reader = sock.makefile("r", encoding="utf-8", newline="\n")
        self.send({"type": "acquire", "client": "tui"})
        raw = self.reader.readline()
        if not raw:
            raise RuntimeError("CORE closed connection during acquire")
        first = json.loads(raw)
        if not isinstance(first, dict):
            raise RuntimeError("invalid CORE acquire response")
        if first.get("type") == "acquired":
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._read_loop,
                name="cat-agent-tui-ipc",
                daemon=True,
            )
            self._thread.start()
        return first

    def send(self, payload: dict[str, object]) -> None:
        sock = self.sock
        if sock is None:
            raise RuntimeError("CORE client is not connected")
        data = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self._send_lock:
            sock.sendall(data)

    def close(self) -> None:
        self._stop.set()
        if self.sock is not None:
            try:
                self.send({"type": "release"})
            except Exception:
                pass
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=0.5)
        if self.reader is not None:
            try:
                self.reader.close()
            except OSError:
                pass
        self._thread = None
        self.reader = None
        self.sock = None

    def _read_loop(self) -> None:
        reader = self.reader
        if reader is None:
            return
        try:
            for raw in reader:
                if self._stop.is_set():
                    return
                line = raw.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    self.incoming.put(item)
        except OSError:
            pass
        finally:
            if not self._stop.is_set():
                self.incoming.put(
                    {"type": "disconnected", "text": "CORE connection closed"}
                )


class TerminalTUI:
    def __init__(
        self,
        client: CoreClient,
        core_info: dict[str, object],
        status: dict[str, object],
    ) -> None:
        self.client = client
        self.core_info = core_info
        self.status = status
        self._dialog: deque[tuple[str, str, int]] = deque(maxlen=1000)
        self._dialog_scroll_lines = 0
        self._dialog_page_lines = 10
        self._input = ""
        self._quit = False
        self._started = time.monotonic()
        self._last_snapshot_request = 0.0
        self._colors_enabled = False

        self._stream_mode = "idle"
        self._stream_buffer = ""
        self._stream_dialog_active = False

    def run(self) -> None:
        self._set_bracketed_paste(True)
        try:
            curses.wrapper(self._main)
        finally:
            self._set_bracketed_paste(False)

    def _main(self, stdscr) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        self._init_colors()
        stdscr.keypad(True)
        stdscr.timeout(100)

        while not self._quit:
            self._request_snapshot()
            self._poll_core()
            self._draw(stdscr)
            try:
                key = stdscr.get_wch()
            except curses.error:
                continue
            if key == "\x1b" and self._try_bracketed_paste(stdscr):
                continue
            self._handle_key(key)

    def _request_snapshot(self) -> None:
        now = time.monotonic()
        if now - self._last_snapshot_request < SNAPSHOT_INTERVAL:
            return
        self._last_snapshot_request = now
        try:
            self.client.send({"type": "snapshot"})
        except OSError:
            pass

    def _poll_core(self) -> None:
        while True:
            try:
                item = self.client.incoming.get_nowait()
            except queue.Empty:
                return

            kind = str(item.get("type", ""))
            if kind == "snapshot":
                snapshot = item.get("status")
                if isinstance(snapshot, dict):
                    self.status = snapshot
                continue
            if kind == "status":
                self.status = {**self.status, **item}
                continue
            if kind == "model_event":
                self._handle_model_event(item)
                continue
            if kind in {"reply", "notification"}:
                self._finalize_manager_text(str(item.get("text", "")))
                continue
            if kind == "busy":
                self._append_dialog("CORE", str(item.get("text", "Гена занят")))
                continue
            if kind == "error":
                self._append_dialog("CORE", str(item.get("error", "CORE error")))
                continue
            if kind == "disconnected":
                self._append_dialog("CORE", str(item.get("text", "CORE disconnected")))
                self._quit = True

    def _handle_model_event(self, item: dict[str, object]) -> None:
        if item.get("label") != "manager":
            return
        event = str(item.get("event", ""))
        payload = str(item.get("payload", ""))
        timing = item.get("timing")
        if isinstance(timing, dict):
            self.status = {**self.status, "inference": timing}

        if event == "decode_start":
            self._stream_mode = "pending"
            self._stream_buffer = ""
            self._stream_dialog_active = False
            return
        if event == "chunk":
            self._feed_stream_chunk(payload)

    def _feed_stream_chunk(self, chunk: str) -> None:
        if not chunk or self._stream_mode == "hidden":
            return
        if self._stream_mode == "visible":
            self._append_stream_text(chunk)
            return

        self._stream_buffer += chunk
        text = self._stream_buffer
        candidates = ("/work#", "ASK\n", "REPLY\n")
        if text.startswith("/work#"):
            self._stream_mode = "hidden"
            self._stream_buffer = ""
            return
        if text.startswith("ASK\n"):
            self._stream_mode = "visible"
            self._stream_buffer = ""
            self._append_stream_text(text[len("ASK\n") :])
            return
        if text.startswith("REPLY\n"):
            self._stream_mode = "visible"
            self._stream_buffer = ""
            self._append_stream_text(text[len("REPLY\n") :])
            return
        if any(candidate.startswith(text) for candidate in candidates):
            return
        if "\n" in text:
            self._stream_mode = "hidden"
            self._stream_buffer = ""

    def _append_stream_text(self, text: str) -> None:
        if not text:
            return
        if not self._stream_dialog_active:
            self._append_dialog("MANAGER", text)
            self._stream_dialog_active = True
            return
        if self._dialog and self._dialog[-1][0] == "MANAGER":
            speaker, current, elapsed = self._dialog.pop()
            self._dialog.append((speaker, current + text, elapsed))

    def _finalize_manager_text(self, text: str) -> None:
        if (
            self._stream_dialog_active
            and self._dialog
            and self._dialog[-1][0] == "MANAGER"
        ):
            speaker, _preview, elapsed = self._dialog.pop()
            self._dialog.append((speaker, text, elapsed))
        else:
            self._append_dialog("MANAGER", text)
        self._stream_mode = "idle"
        self._stream_buffer = ""
        self._stream_dialog_active = False
        self._dialog_scroll_lines = 0

    def _append_dialog(self, speaker: str, text: str) -> None:
        elapsed = max(0, int(time.monotonic() - self._started))
        self._dialog.append((speaker, text, elapsed))
        self._dialog_scroll_lines = 0

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
                0, self._dialog_scroll_lines - self._dialog_page_lines
            )
            return
        if key == curses.KEY_END:
            self._dialog_scroll_lines = 0
            return
        if key == "\x03":
            self._quit = True
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
        self._append_dialog("YOU", text)
        try:
            self.client.send({"type": "user", "text": text})
        except OSError as exc:
            self._append_dialog("CORE", f"send failed: {exc}")

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
            lines.extend((indent, part, False) for part in wrapped[1:])
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
            self._safe_addstr(
                win,
                0,
                max(2, width - len(marker) - 2),
                marker,
                curses.A_BOLD,
            )

    def _draw_info(self, win) -> None:
        height, width = win.getmaxyx()
        row = 1

        def put(label: str, value: object = "", *, bold: bool = False) -> None:
            nonlocal row
            if row >= height - 1:
                return
            text = f"{label:<12} {value}" if value != "" else label
            self._safe_addstr(
                win,
                row,
                2,
                text,
                curses.A_BOLD if bold else 0,
                width - 4,
            )
            row += 1

        def put_indicator(label: str, value: str, attr: int) -> None:
            nonlocal row
            if row >= height - 1:
                return
            lead = f"{label:<12} "
            self._safe_addstr(win, row, 2, lead, 0, width - 4)
            x = 2 + len(lead)
            if x < width - 2:
                self._safe_addstr(win, row, x, "●", attr | curses.A_BOLD, 1)
                self._safe_addstr(
                    win,
                    row,
                    x + 2,
                    value,
                    0,
                    width - x - 4,
                )
            row += 1

        state = str(self.status.get("state", "IDLE"))
        label = str(self.status.get("label", ""))
        put("STATE", f"{state} {label}".strip(), bold=True)

        chat_open = bool(self.status.get("chat_open"))
        put_indicator(
            "Чат",
            "открыт" if chat_open else "закрыт",
            self._indicator_attr(chat_open),
        )

        started = self.status.get("request_started_monotonic")
        last = self.status.get("last_request_seconds")
        if isinstance(started, (int, float)):
            put("request", f"{time.monotonic() - float(started):.1f}s")
        elif isinstance(last, (int, float)):
            put("request", f"{float(last):.3f}s")
        else:
            put("request", "--")

        timing = self.status.get("inference")
        if not isinstance(timing, dict):
            timing = {}
        phase = str(timing.get("phase", "idle"))
        phase_started = timing.get("phase_started")
        prefill = timing.get("prefill_seconds")
        generation = timing.get("generation_seconds")
        total = timing.get("total_seconds")
        if phase == "prefill" and isinstance(phase_started, (int, float)):
            prefill = time.monotonic() - float(phase_started)
            generation = None
            total = None
        elif phase == "generate" and isinstance(phase_started, (int, float)):
            generation = time.monotonic() - float(phase_started)
            total = None
        put("prefill", self._seconds(prefill))
        put("generate", self._seconds(generation))
        put("total", self._seconds(total))

        row += 1
        put("MODEL", self.core_info.get("model", "?"), bold=True)
        put("backend", self.core_info.get("backend", "?"))
        put("speculative", "ON" if self.core_info.get("speculative") else "OFF")
        put(
            "engine M/A",
            f"{self._seconds(self.core_info.get('manager_engine_seconds'))} / "
            f"{self._seconds(self.core_info.get('agent_engine_seconds'))}",
        )

        row += 1
        put("WARM", bold=True)
        put(
            "manager",
            f"{self.core_info.get('manager_warm_tokens', '?')} tok "
            f"{self._seconds(self.core_info.get('manager_warm_seconds'))}",
        )
        put(
            "agent",
            f"{self.core_info.get('agent_warm_tokens', '?')} tok "
            f"{self._seconds(self.core_info.get('agent_warm_seconds'))}",
        )

        row += 1
        put("MANAGER", bold=True)
        row = self._draw_client_stats(
            win,
            row,
            self.status.get("manager"),
            width,
            height,
        )

        row += 1
        put("AGENT", bold=True)
        row = self._draw_client_stats(
            win,
            row,
            self.status.get("agent"),
            width,
            height,
        )

        row += 1
        put("EVENTS", bold=True)

        tasks = self.status.get("tasks")
        task_items = (
            [item for item in tasks if isinstance(item, dict)]
            if isinstance(tasks, list)
            else []
        )
        timer_active = any(
            self._task_enabled(item) and isinstance(item.get("timer"), dict)
            for item in task_items
        )
        external_active = any(
            self._task_enabled(item) and not isinstance(item.get("timer"), dict)
            for item in task_items
        )

        source_lead = f"{'sources':<12} "
        self._safe_addstr(win, row, 2, source_lead, 0, width - 4)
        source_x = 2 + len(source_lead)
        for text, active in (
            (" timer", timer_active),
            (" gpio", False),
            (" mqtt", external_active),
        ):
            dot = "●" if source_x == 2 + len(source_lead) else " ●"
            attr = self._event_indicator_attr(active)
            if source_x >= width - 2:
                break
            self._safe_addstr(win, row, source_x, dot, attr, width - source_x - 2)
            source_x += len(dot)
            self._safe_addstr(win, row, source_x, text, attr, width - source_x - 2)
            source_x += len(text)
        row += 1

        if not task_items:
            put("tasks", "none")
        else:
            now = time.monotonic()
            for raw_task in task_items:
                if row >= height - 1:
                    break
                task_id = raw_task.get("task_id", "?")
                description = str(raw_task.get("description", ""))
                timer = raw_task.get("timer")
                if not isinstance(timer, dict):
                    enabled = self._task_enabled(raw_task)
                    put_indicator(
                        f"TASK {task_id}",
                        "EXTERNAL",
                        self._event_indicator_attr(enabled),
                    )
                else:
                    period = timer.get("period_seconds")
                    enabled = bool(timer.get("enabled"))
                    next_fire = timer.get("next_fire_monotonic")
                    period_text = self._number(period)
                    if enabled and isinstance(next_fire, (int, float)):
                        next_in = max(0.0, float(next_fire) - now)
                        put_indicator(
                            f"TASK {task_id}",
                            f"RUN {period_text}s  next {next_in:.0f}s",
                            self._event_indicator_attr(True),
                        )
                    else:
                        put_indicator(
                            f"TASK {task_id}",
                            f"STOP {period_text}s",
                            self._event_indicator_attr(False),
                        )
                if row < height - 1:
                    self._safe_addstr(
                        win,
                        row,
                        15,
                        description,
                        curses.A_DIM,
                        max(0, width - 17),
                    )
                    row += 1

        pending = self.status.get("pending")
        if isinstance(pending, int) and pending:
            put("queue", str(pending))

        row += 1
        put("LOG", bold=True)
        put("file", str(LOG_PATH))
        try:
            put("size", self._format_bytes(LOG_PATH.stat().st_size))
        except OSError:
            put("size", "--")

    def _draw_client_stats(
        self,
        win,
        row: int,
        raw_client: object,
        width: int,
        height: int,
    ) -> int:
        def put(text: str) -> None:
            nonlocal row
            if row >= height - 1:
                return
            self._safe_addstr(win, row, 2, text, 0, width - 4)
            row += 1

        client = raw_client if isinstance(raw_client, dict) else {}
        put(f"resident     {client.get('resident_tokens', '?')}")
        response = client.get("last")
        if not isinstance(response, dict):
            put("last         --")
            return row

        cached = response.get("cached_tokens")
        new = response.get("prompt_evaluated_tokens")
        decode = response.get("completion_tokens")
        put(f"cached/new   {cached or 0} / {new or 0}")
        put(f"decode       {decode or 0}")
        put(f"prefill      {self._seconds(response.get('prompt_seconds'))}")
        put(f"generate     {self._seconds(response.get('generation_seconds'))}")
        put(f"wall         {self._seconds(response.get('elapsed_seconds'))}")
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
            pasted = " ".join(data.replace("\r", "\n").splitlines()).strip()
            if pasted:
                self._input += pasted
            return True
        finally:
            stdscr.timeout(100)

    def _init_colors(self) -> None:
        if not curses.has_colors():
            return
        try:
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(COLOR_ACTIVE, curses.COLOR_GREEN, -1)
            curses.init_pair(COLOR_STOPPED, curses.COLOR_RED, -1)
        except curses.error:
            return
        self._colors_enabled = True

    def _indicator_attr(self, active: bool) -> int:
        if not self._colors_enabled:
            return curses.A_BOLD if active else curses.A_DIM
        pair = COLOR_ACTIVE if active else COLOR_STOPPED
        return curses.color_pair(pair)

    def _event_indicator_attr(self, active: bool) -> int:
        return self._indicator_attr(True) if active else curses.A_DIM

    @staticmethod
    def _task_enabled(task: dict[str, object]) -> bool:
        if "enabled" in task:
            return bool(task.get("enabled"))
        timer = task.get("timer")
        if isinstance(timer, dict):
            return bool(timer.get("enabled"))
        return True

    @staticmethod
    def _set_bracketed_paste(enabled: bool) -> None:
        sequence = BRACKETED_PASTE_ON if enabled else BRACKETED_PASTE_OFF
        try:
            sys.stdout.write(sequence)
            sys.stdout.flush()
        except OSError:
            pass

    @staticmethod
    def _seconds(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}s"
        return "--"

    @staticmethod
    def _number(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):g}"
        return "?"

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
        TerminalTUI._safe_addstr(win, 0, 0, top)
        for row in range(1, height - 1):
            TerminalTUI._safe_addstr(win, row, 0, "│")
            TerminalTUI._safe_addstr(win, row, width - 1, "│")
        TerminalTUI._safe_addstr(win, height - 1, 0, bottom)

    @staticmethod
    def _title(win, text: str) -> None:
        try:
            win.addstr(0, 2, text, curses.A_BOLD)
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


def main() -> int:
    raw_path = os.getenv("CAT_AGENT_CORE_SOCKET", str(DEFAULT_CORE_SOCKET)).strip()
    path = Path(raw_path)
    client = CoreClient(path)
    try:
        try:
            first = client.connect()
        except OSError as exc:
            print(f"Cannot connect to cat-agent CORE at {path}: {exc}", file=sys.stderr)
            return 2

        if first.get("type") == "busy":
            owner = first.get("owner") or "unknown"
            print(f"Гена занят ({owner})")
            return 3
        if first.get("type") != "acquired":
            print(f"Unexpected CORE response: {first}", file=sys.stderr)
            return 2

        core_info = first.get("core")
        status = first.get("status")
        TerminalTUI(
            client,
            core_info if isinstance(core_info, dict) else {},
            status if isinstance(status, dict) else {},
        ).run()
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())