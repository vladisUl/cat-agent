from __future__ import annotations

import curses
import queue
import time

from .priority_tui import PriorityLiteRTTUI


class LivePriorityLiteRTTUI(PriorityLiteRTTUI):
    """Priority TUI with live prefill/generate timing and visible response streaming."""

    def __init__(self, bundle) -> None:
        super().__init__(bundle)
        self._model_events: queue.SimpleQueue[tuple[str, str, str]] = queue.SimpleQueue()
        self._stream_label: str | None = None
        self._stream_mode = "idle"
        self._stream_buffer = ""
        self._stream_dialog_active = False

        # Only model calls that can directly produce user-visible text need
        # streamed decode. Agent tool traffic stays on the blocking Session path.
        self.bundle.manager_client.set_event_handler(self._queue_model_event)
        self.bundle.direct_client.set_event_handler(self._queue_model_event)

    def _queue_model_event(self, label: str, event: str, payload: str) -> None:
        self._model_events.put((label, event, payload))

    def _draw(self, stdscr) -> None:
        self._poll_model_events()
        super()._draw(stdscr)

    def _poll_model_events(self) -> None:
        while True:
            try:
                label, event, payload = self._model_events.get_nowait()
            except queue.Empty:
                return

            if event == "decode_start":
                self._stream_label = label
                self._stream_mode = "pending"
                self._stream_buffer = ""
                continue

            if event == "chunk":
                self._feed_stream_chunk(label, payload)
                continue

            if event in {"decode_done", "decode_error"}:
                # Completion is finalized when the orchestration future returns;
                # until then keep the streamed dialog entry alive.
                continue

    def _feed_stream_chunk(self, label: str, chunk: str) -> None:
        if not chunk:
            return
        if label != self._stream_label:
            self._stream_label = label
            self._stream_mode = "pending"
            self._stream_buffer = ""

        if self._stream_mode == "hidden":
            return
        if self._stream_mode == "visible":
            self._append_stream_text(chunk)
            return

        self._stream_buffer += chunk
        if label == "manager":
            self._classify_manager_stream()
        elif label == "manager-direct":
            self._classify_direct_stream()
        else:
            self._stream_mode = "hidden"
            self._stream_buffer = ""

    def _classify_manager_stream(self) -> None:
        if "\n" not in self._stream_buffer:
            return
        first, _sep, rest = self._stream_buffer.partition("\n")
        control = first.strip()
        self._stream_buffer = ""
        if control in {"REPLY", "ASK"}:
            self._stream_mode = "visible"
            self._append_stream_text(rest)
        else:
            self._stream_mode = "hidden"

    def _classify_direct_stream(self) -> None:
        candidates = ("/work#", "ASK\n", "REPLY\n")
        text = self._stream_buffer

        if text.startswith("/work#"):
            self._stream_mode = "hidden"
            self._stream_buffer = ""
            return
        if text.startswith("ASK\n"):
            self._stream_mode = "visible"
            self._stream_buffer = ""
            self._append_stream_text(text[len("ASK\n"):])
            return
        if text.startswith("REPLY\n"):
            self._stream_mode = "visible"
            self._stream_buffer = ""
            self._append_stream_text(text[len("REPLY\n"):])
            return
        if any(candidate.startswith(text) for candidate in candidates):
            return

        self._stream_mode = "visible"
        self._stream_buffer = ""
        self._append_stream_text(text)

    def _append_stream_text(self, text: str) -> None:
        if not text:
            return
        if not self._stream_dialog_active:
            self._append_dialog("MANAGER", text)
            self._stream_dialog_active = True
        else:
            if self._dialog and self._dialog[-1][0] == "MANAGER":
                speaker, current, elapsed = self._dialog.pop()
                self._dialog.append((speaker, current + text, elapsed))
            else:
                self._append_dialog("MANAGER", text)
        self._dialog_scroll_lines = 0

    def _poll_future(self) -> None:
        previous_future = self._active_future
        super()._poll_future()
        if previous_future is not None and self._active_future is None:
            self._finalize_stream_dialog()

    def _finalize_stream_dialog(self) -> None:
        if not self._stream_dialog_active:
            self._stream_label = None
            self._stream_mode = "idle"
            self._stream_buffer = ""
            return

        # Base TUI appends the authoritative ManagerTurn when the future ends.
        # Remove the streamed preview immediately before it so the chat contains
        # one final message, not a streamed copy plus a duplicate.
        if len(self._dialog) >= 2:
            last = self._dialog[-1]
            previous = self._dialog[-2]
            if last[0] == "MANAGER" and previous[0] == "MANAGER":
                last = self._dialog.pop()
                previous = self._dialog.pop()
                self._dialog.append((last[0], last[1], previous[2]))

        self._stream_dialog_active = False
        self._stream_label = None
        self._stream_mode = "idle"
        self._stream_buffer = ""
        self._dialog_scroll_lines = 0

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

        put("STATE", self._status, bold=True)
        timing = self._current_inference_timing()
        if timing is None:
            put("prefill", "--")
            put("generate", "--")
            put("total", "--")
        else:
            now = time.monotonic()
            if timing.phase == "prefill" and timing.phase_started is not None:
                prefill = now - timing.phase_started
                generate = None
                total = None
            elif timing.phase == "generate" and timing.phase_started is not None:
                prefill = timing.prefill_seconds
                generate = now - timing.phase_started
                total = None
            else:
                prefill = timing.prefill_seconds
                generate = timing.generation_seconds
                total = timing.total_seconds
            put("prefill", self._seconds(prefill))
            put("generate", self._seconds(generate))
            put("total", self._seconds(total))

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
        if self.bundle.direct_warm is not None:
            put(
                "direct",
                f"{self.bundle.direct_warm.token_count} tok "
                f"{self.bundle.direct_warm.elapsed_seconds:.3f}s",
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
        put("DIRECT", bold=True)
        row = self._draw_client_stats(win, row, self.bundle.direct_client, width, height)

        row += 1
        put("AGENT", bold=True)
        row = self._draw_client_stats(win, row, self.bundle.agent_client, width, height)

        row += 1
        put("EVENTS", bold=True)
        source_lead = f"{'sources':<12} "
        self._safe_addstr(win, row, 2, source_lead, 0, width - 4)
        source_x = 2 + len(source_lead)
        active_attr = self._indicator_attr(True)
        stub_attr = curses.A_DIM
        for dot, text, attr in (
            ("●", " timer", active_attr),
            (" ·", " gpio", stub_attr),
            (" ·", " mqtt", stub_attr),
        ):
            if source_x >= width - 2:
                break
            self._safe_addstr(win, row, source_x, dot, attr, width - source_x - 2)
            source_x += len(dot)
            self._safe_addstr(win, row, source_x, text, attr, width - source_x - 2)
            source_x += len(text)
        row += 1

        tasks = self.bundle.system_runtime.task_snapshot()
        timers = {
            timer.task_id: timer
            for timer in self.bundle.system_runtime.task_timer_snapshot()
        }
        if not tasks:
            put("tasks", "none")
        else:
            now = time.monotonic()
            for task in tasks:
                if row >= height - 1:
                    break
                timer = timers.get(task.task_id)
                if timer is None:
                    put_indicator(f"TASK {task.task_id}", "NO TIMER", curses.A_DIM)
                elif timer.enabled and timer.next_fire_monotonic is not None:
                    next_in = max(0.0, timer.next_fire_monotonic - now)
                    put_indicator(
                        f"TASK {task.task_id}",
                        f"RUN {timer.period_seconds:g}s  next {next_in:.0f}s",
                        self._indicator_attr(True),
                    )
                else:
                    put_indicator(
                        f"TASK {task.task_id}",
                        f"STOP {timer.period_seconds:g}s",
                        self._indicator_attr(False),
                    )
                if row < height - 1:
                    self._safe_addstr(
                        win,
                        row,
                        15,
                        task.description,
                        curses.A_DIM,
                        max(0, width - 17),
                    )
                    row += 1

        if self._pending:
            put("queue", str(len(self._pending)))

    def _current_inference_timing(self):
        clients = [
            self.bundle.manager_client,
            self.bundle.direct_client,
            *self.bundle.agent_clients,
        ]
        for client in clients:
            timing = client.inference_timing
            if timing.phase != "idle":
                return timing

        finished = [
            client.inference_timing
            for client in clients
            if client.inference_timing.finished_at is not None
        ]
        if not finished:
            return None
        return max(finished, key=lambda item: item.finished_at or 0.0)
