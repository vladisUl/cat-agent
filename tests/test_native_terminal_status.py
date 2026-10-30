"""Native TUI status: separate KV prefaces and authoritative timer countdowns."""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from agent_core_tools.native_terminal import NativeTerminalTUI


class FakeWindow:
    def __init__(self, height=45, width=75):
        self.height = height
        self.width = width
        self.writes = []

    def getmaxyx(self):
        return self.height, self.width

    def addnstr(self, y, x, text, n, attr=0):
        self.writes.append((y, str(text)[:n]))

    def addstr(self, y, x, text, attr=0):
        self.writes.append((y, str(text)))

    @property
    def content(self):
        return " ".join(text for _, text in self.writes)


class NativeTuiStatusTests(unittest.TestCase):
    def make_tui(self, *, armed=True):
        tui = NativeTerminalTUI.__new__(NativeTerminalTUI)
        tui._colors_enabled = False
        tui._pending_query_ack = {}
        tui.core_info = {
            "model": "gemma-4-E2B-it.litertlm",
            "backend": "GPU",
            "stage": 5,
            "manager_warm_tokens": 1166,
        }
        tui.status = {
            "chat_open": True,
            "state": "IDLE",
            "label": "",
            "pending": 0,
            "agents": [{"agent_id": "agent_1", "state": "FREE",
                         "base_kv_tokens": 418}],
            "tasks": [{
                "task_id": 2, "executor": "litert", "method": "query",
                "description": "LiteRT QUERY: get_outdoor_weather",
                "enabled": armed,
                "timer": {"enabled": armed, "period_seconds": 60.0,
                          "next_fire_monotonic": 1045.0 if armed else None},
            }],
            "last_agent_runs": {
                "agent_1": {"task_id": 2, "outcome": "done", "at_wall": 0.0},
            },
            "task_worker": {"ready": True, "reserved_agents": [],
                            "offered": 0, "unacknowledged": 0,
                            "pending_query_validations": 0},
        }
        return tui

    def test_running_task_countdown_separate_base_and_last_run(self):
        tui = self.make_tui()
        win = FakeWindow()
        with patch("agent_core_tools.native_terminal.time.monotonic",
                   return_value=1000.0):
            tui._draw_info(win)
        text = win.content
        self.assertIn("backend", text)
        self.assertIn("GPU", text)
        self.assertIn("manager", text)
        self.assertIn("1166 tok", text)
        self.assertIn("FREE  KV 418", text)
        self.assertIn("TASK 2 done", text)
        self.assertIn("RUN 60s  next 45s", text)
        self.assertIn("LiteRT QUERY: get_outdoor_weather", text)

    def test_stopped_task_does_not_show_running_countdown(self):
        tui = self.make_tui(armed=False)
        win = FakeWindow()
        tui._draw_info(win)
        self.assertIn("STOP 60s", win.content)
        self.assertNotIn("next 45s", win.content)

    def test_task_status_error_is_not_misreported_as_no_tasks(self):
        tui = self.make_tui()
        tui.status["tasks"] = []
        tui.status["task_display_error"] = "TASK SYSTEM unavailable"
        win = FakeWindow()
        tui._draw_info(win)
        self.assertIn("tasks", win.content)
        self.assertIn("unavailable", win.content)
        self.assertNotIn("RUN 60s", win.content)


if __name__ == "__main__":
    unittest.main()
