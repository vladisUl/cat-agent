"""Stage 4B: diagnostic hardware queue, bounded job status, slice time limits.

Uses the stage-4A FakeManager and its deterministic blocked first slice.
No LiteRT model, MQTT broker or Task SYSTEM is required.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_core_tools.server import CoreServer
from orchestration_tools.agents import NativeAgent
from test_agent_core_tools_priority import FakeManager, FakeAgentChat


class HardwareAndLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager = FakeManager()
        # Two independent Conversations: agent_1 background; agent_2 hardware.
        self.manager.pool._agents["agent_2"] = NativeAgent(
            "agent_2", FakeAgentChat(
                threading.Event(), threading.Event(), self.manager.events,
            ),
        )
        self.manager.run_agent = lambda ident, task: self.manager.pool.run(task, ident)
        self.server = CoreServer(
            Path(self.tmp.name) / "core.sock", manager=self.manager,
            background_enabled=True, hardware_probe_enabled=True,
        )
        self.server.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.settimeout(3)
        self.conn.connect(str(self.server.path))
        self.reader = self.conn.makefile("r", encoding="utf-8")
        self.addCleanup(self.reader.close)
        self.addCleanup(self.conn.close)
        self.assertEqual(self.query({"type": "acquire", "client": "test"})["type"], "acquired")

    def shutdown(self):
        self.manager.resume.set()
        self.server.close()
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())

    def send(self, payload):
        self.conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode())

    def read(self):
        line = self.reader.readline()
        self.assertTrue(line, "IPC disconnected")
        return json.loads(line)

    def query(self, payload):
        self.send(payload)
        return self.read()

    def until(self, target_type):
        for _ in range(30):
            message = self.read()
            if message["type"] == target_type:
                return message
        self.fail("no " + target_type)

    def test_human_over_hardware_over_background_and_agent_kv_separation(self):
        accepted = self.query({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2", "step3"],
        })
        self.assertEqual(accepted["type"], "background_accepted")
        self.assertTrue(self.manager.entered.wait(2))
        hw = self.query({
            "type": "hardware_probe", "source": "gpio", "name": "door",
            "agent": "agent_2", "text": "hardware-action",
        })
        self.assertEqual(hw["type"], "hardware_accepted")
        self.assertTrue(hw["event_id"])
        human = self.query({"type": "user", "text": "urgent"})
        self.assertEqual(human["type"], "accepted")
        self.manager.resume.set()
        received = []
        while not any(m["type"] == "background_completed" for m in received):
            received.append(self.read())
        self.assertTrue(any(m["type"] == "hardware_completed" and m["ok"]
                            for m in received))
        self.assertTrue(any(m["type"] == "completed" for m in received))
        self.assertEqual(self.manager.events[:5],
                         ["step1", "human:urgent", "hardware-action", "step2", "step3"])
        self.assertEqual(self.manager.agent_chat.reset_count, 1)
        status = self.query({
            "type": "background_status", "background_id": accepted["background_id"],
        })
        self.assertEqual(status["job"]["state"], "COMPLETED")
        self.assertEqual(status["job"]["completed_steps"], 3)

    def test_timeout_between_native_turns_not_mid_inference(self):
        accepted = self.query({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
            "max_runtime_seconds": 1,
        })
        self.assertTrue(self.manager.entered.wait(2))
        # Force expiry while step1 is already in LiteRT. The running step
        # must finish; step2 must never start. No wall-clock sleeps needed.
        with self.server._lock:
            self.server._backgrounds[accepted["background_id"]].started_at -= 3600
        self.manager.resume.set()
        completed = self.until("background_completed")
        self.assertEqual(completed["state"], "TIMED_OUT")
        self.assertFalse(completed["ok"])
        self.assertNotIn("step2", self.manager.events)
        self.assertEqual(self.manager.agent_chat.working_kv_tokens, 200)
        status = self.query({
            "type": "background_status", "background_id": accepted["background_id"],
        })
        self.assertEqual(status["job"]["state"], "TIMED_OUT")

    def test_cancel_terminal_query_and_new_agent_task(self):
        accepted = self.query({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        cancel = self.query({
            "type": "cancel_background", "background_id": accepted["background_id"],
        })
        self.assertTrue(cancel["cancelled"])
        self.manager.resume.set()
        finished = self.until("background_completed")
        self.assertEqual(finished["state"], "CANCELLED")
        self.assertNotIn("step2", self.manager.events)
        status = self.query({
            "type": "background_status", "background_id": accepted["background_id"],
        })
        self.assertEqual(status["job"]["state"], "CANCELLED")
        next_task = self.query({
            "type": "agent_task", "agent": "agent_1", "text": "fresh",
        })
        self.assertEqual(next_task["type"], "accepted")
        self.assertEqual(self.until("completed")["kind"], "reply")

    def test_hardware_cannot_use_suspended_background_agent(self):
        accepted = self.query({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        response = self.query({
            "type": "hardware_probe", "source": "gpio", "name": "door",
            "agent": "agent_1", "text": "hardware-action",
        })
        self.assertEqual(response["type"], "error")
        self.assertIn("background job", response["error"])
        self.manager.resume.set()
        self.until("background_completed")

    def test_probe_rejects_task_run_spoof_and_unowned_clients(self):
        bogus = self.query({
            "type": "hardware_probe", "source": "gpio", "name": "door",
            "agent": "agent_2", "text": "test", "run_id": "fake",
        })
        self.assertEqual(bogus["type"], "error")
        with socket.socket(socket.AF_UNIX) as other:
            other.settimeout(2)
            other.connect(str(self.server.path))
            with other.makefile("r", encoding="utf-8") as reader:
                other.sendall((json.dumps({
                    "type": "hardware_probe", "source": "gpio", "name": "door",
                    "agent": "agent_2", "text": "test",
                })+"\n").encode())
                self.assertEqual(json.loads(reader.readline())["type"], "busy")
        self.assertEqual(self.manager.events, [])

    def test_failed_kv_reset_is_reported_as_failed_job(self):
        accepted = self.query({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        self.assertTrue(self.query({
            "type": "cancel_background",
            "background_id": accepted["background_id"],
        })["cancelled"])
        with patch.object(self.manager.agent_chat, "new_dialogue",
                          side_effect=RuntimeError("BASE clone failed")):
            self.manager.resume.set()
            terminal = self.until("background_completed")
        self.assertEqual(terminal["state"], "FAILED")
        self.assertIn("BASE clone failed", terminal["text"])
        status = self.query({
            "type": "background_status",
            "background_id": accepted["background_id"],
        })
        self.assertEqual(status["job"]["state"], "FAILED")
        self.assertEqual(self.manager.pool.status()[0]["state"], "FAILED")
        self.assertNotIn("step2", self.manager.events)

    def test_finished_job_is_not_visible_after_session_epoch_change(self):
        accepted = self.query({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        self.manager.resume.set()
        self.until("background_completed")
        old = self.query({
            "type": "background_status", "background_id": accepted["background_id"],
        })
        self.assertEqual(old["job"]["state"], "COMPLETED")
        self.assertEqual(self.query({"type": "release"})["type"], "released")
        self.assertEqual(self.query({"type": "acquire", "client": "again"})["type"],
                         "acquired")
        new = self.query({
            "type": "background_status", "background_id": accepted["background_id"],
        })
        self.assertIsNone(new["job"])

    def test_invalid_wall_time_rejected_without_side_effect(self):
        for runtime in (0, -1, 3601, True, float("nan"), "invalid"):
            with self.subTest(runtime=runtime):
                response = self.query({
                    "type": "background_task", "agent": "agent_1",
                    "steps": ["x", "y"], "max_runtime_seconds": runtime,
                })
                self.assertEqual(response["type"], "error")
        self.assertEqual(self.manager.events, [])
        self.assertEqual(self.query({"type": "snapshot"})["status"]["background_jobs"], [])


if __name__ == "__main__":
    unittest.main()
