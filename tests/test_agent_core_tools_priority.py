"""Stage 4A: native Engine priority scheduling (no real LiteRT required)."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_core_tools.priority_scheduler import PriorityScheduler, QueueFullError
from agent_core_tools.server import CoreServer
from orchestration_tools.agents import AgentPool, NativeAgent
from orchestration_tools.contracts import ManagerTurn


class SchedulerTests(unittest.TestCase):
    def test_human_precedes_waiting_background_and_fifo(self):
        scheduler = PriorityScheduler()
        self.addCleanup(scheduler.close)
        entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
        order = []

        def slow():
            order.append("background1")
            entered.set()
            self.assertTrue(resume.wait(3))

        scheduler.submit("background", slow)
        self.assertTrue(entered.wait(2))
        scheduler.submit("background", lambda: order.append("background2"))
        scheduler.submit("hardware", lambda: order.append("hardware"))
        scheduler.submit("human", lambda: order.append("human1"))
        scheduler.submit("human", lambda: order.append("human2"))
        scheduler.submit("background", lambda: (order.append("background3"), finished.set()))
        self.assertEqual(scheduler.snapshot()["pending"], 5)
        resume.set()
        self.assertTrue(finished.wait(2))
        self.assertEqual(order, [
            "background1", "human1", "human2", "hardware",
            "background2", "background3",
        ])

    def test_control_reset_first_and_admission_boundary(self):
        scheduler = PriorityScheduler(max_pending=1)
        self.addCleanup(scheduler.close)
        entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(resume.set)  # Free the blocked worker even if an assertion fails.
        order = []

        def slow():
            entered.set()
            self.assertTrue(resume.wait(3))

        scheduler.submit("background", slow)
        self.assertTrue(entered.wait(2))
        scheduler.submit("human", lambda: (order.append("human"), finished.set()))
        with self.assertRaises(QueueFullError):
            scheduler.submit("background", lambda: None)
        # Model-side context reset must not be rejected when user queue is full.
        scheduler.submit("control", lambda: order.append("reset"))
        resume.set()
        self.assertTrue(finished.wait(2))
        self.assertEqual(order, ["reset", "human"])

    def test_admitted_background_continuation_survives_queue_pressure(self):
        scheduler = PriorityScheduler(max_pending=1)
        self.addCleanup(scheduler.close)
        entered, resume, finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(resume.set)  # Free the blocked worker even if an assertion fails.
        order = []

        def slow():
            entered.set()
            self.assertTrue(resume.wait(3))

        scheduler.submit("background", slow)
        self.assertTrue(entered.wait(2))
        scheduler.submit("human", lambda: order.append("human"))
        scheduler.submit("background", lambda: (order.append("continuation"), finished.set()),
                         continuation=True)
        with self.assertRaises(QueueFullError):
            scheduler.submit("human", lambda: None)
        resume.set()
        self.assertTrue(finished.wait(2))
        self.assertEqual(order, ["human", "continuation"])

    def test_close_never_executes_pending_side_effects(self):
        scheduler = PriorityScheduler()
        entered, resume = threading.Event(), threading.Event()
        order = []

        def slow():
            entered.set()
            self.assertTrue(resume.wait(3))

        scheduler.submit("background", slow)
        self.assertTrue(entered.wait(2))
        scheduler.submit("background", lambda: order.append("replayed"))
        closer = threading.Thread(target=scheduler.close)
        closer.start()
        time.sleep(.05)
        resume.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.assertEqual(order, [])


class FakeAgentChat:
    base_kv_tokens = 200

    def __init__(self, entered, resume, events):
        self.kv = 200
        self.history = []
        self.reset_count = 0
        self.entered, self.resume, self.events = entered, resume, events

    @property
    def working_kv_tokens(self):
        return self.kv

    def ask(self, question):
        self.history.append(question)
        self.kv += 30
        self.events.append(question)
        if question == "step1":
            self.entered.set()
            if not self.resume.wait(4):
                raise TimeoutError("test step timed out")
        return " / ".join(self.history)

    def new_dialogue(self):
        self.kv = self.base_kv_tokens
        self.history.clear()
        self.reset_count += 1

    def close(self):
        pass


class FakeManager:
    base_kv_tokens = 300
    model_name = "gemma"

    def __init__(self):
        self.events = []
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.agent_chat = FakeAgentChat(self.entered, self.resume, self.events)
        self.pool = AgentPool([NativeAgent("agent_1", self.agent_chat)])
        self.closed = False

    def agent_status(self):
        return self.pool.status()

    def run_agent_slice(self, agent_id, text, *, final):
        return self.pool.run_slice(agent_id, text, final=final)

    def abort_agent_slices(self, agent_id):
        self.pool.abort_slices(agent_id)
        self.events.append("abort")

    def respond(self, session_id, text):
        self.events.append("human:" + text)
        return ManagerTurn("reply", "human says " + text)

    def release_session(self, session_id):
        self.events.append("release")

    def close(self):
        self.closed = True
        self.pool.close()


class BackgroundIpcTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manager = FakeManager()
        self.server = CoreServer(Path(self.tmp.name) / "test.sock", manager=self.manager,
                                 background_enabled=True)
        self.server.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.shutdown)
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.conn.settimeout(5)
        self.conn.connect(str(self.server.path))
        self.reader = self.conn.makefile("r", encoding="utf-8")
        self.addCleanup(self.reader.close)
        self.addCleanup(self.conn.close)
        self.assertEqual(self.request({"type": "acquire", "client": "test"})["type"], "acquired")

    def shutdown(self):
        self.manager.resume.set()
        self.server.close()
        self.thread.join(2)
        self.assertFalse(self.thread.is_alive())

    def send(self, payload):
        self.conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode())

    def receive(self):
        raw = self.reader.readline()
        self.assertTrue(raw)
        return json.loads(raw)

    def request(self, payload):
        self.send(payload)
        return self.receive()

    def test_interactive_reply_between_two_agent_slices(self):
        accepted = self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2", "step3"],
        })
        self.assertEqual(accepted["type"], "background_accepted")
        self.assertTrue(self.manager.entered.wait(2))
        self.send({"type": "user", "text": "urgent"})
        self.assertEqual(self.receive()["type"], "accepted")
        self.manager.resume.set()
        received = []
        while not any(p["type"] == "background_completed" for p in received):
            received.append(self.receive())
        types = [item["type"] for item in received]
        self.assertEqual(types.count("completed"), 1)
        self.assertEqual(types.count("background_progress"), 3)
        self.assertEqual(types.count("background_completed"), 1)
        self.assertTrue(received[-1]["ok"])
        self.assertEqual(self.manager.events[:4], [
            "step1", "human:urgent", "step2", "step3",
        ])
        self.assertEqual(self.manager.agent_chat.reset_count, 1)
        self.assertEqual(self.manager.agent_chat.working_kv_tokens, 200)
        self.assertEqual(self.request({"type": "snapshot"})["status"]["background_jobs"], [])

    def test_release_cancels_unstarted_slices_without_replay(self):
        accepted = self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertEqual(accepted["type"], "background_accepted")
        self.assertTrue(self.manager.entered.wait(2))
        self.assertEqual(self.request({"type": "release"})["type"], "released")
        self.manager.resume.set()
        # Wait until model-side cleanup has run on the scheduler.
        for _ in range(100):
            if self.manager.agent_chat.reset_count:
                break
            time.sleep(.01)
        self.assertEqual(self.manager.agent_chat.reset_count, 1)
        self.assertNotIn("step2", self.manager.events)

    def test_background_rejects_second_job_for_same_agent(self):
        self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        error = self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertEqual(error["type"], "error")
        self.assertIn("already", error["error"])
        self.manager.resume.set()

    def test_old_queued_human_request_cannot_run_after_reacquire(self):
        accepted = self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        self.send({"type": "user", "text": "old-user-request"})
        self.assertEqual(self.receive()["type"], "accepted")
        self.assertEqual(self.request({"type": "release"})["type"], "released")
        self.assertEqual(self.request({"type": "acquire", "client": "again"})["type"],
                         "acquired")
        self.send({"type": "user", "text": "new-user-request"})
        self.assertEqual(self.receive()["type"], "accepted")
        self.manager.resume.set()
        messages = []
        while not any(m["type"] == "completed" for m in messages):
            messages.append(self.receive())
        self.assertEqual(next(m for m in messages if m["type"] == "reply")["text"],
                         "human says new-user-request")
        self.assertNotIn("human:old-user-request", self.manager.events)
        self.assertEqual(self.manager.events.count("human:new-user-request"), 1)
        self.assertNotIn("step2", self.manager.events)

    def test_write_failure_cancels_future_background_steps(self):
        accepted = self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertEqual(accepted["type"], "background_accepted")
        self.assertTrue(self.manager.entered.wait(2))
        client = self.server._human
        original_send = client.send
        background_writes = []

        def send_with_disconnected_reader(payload):
            # Track unsolicited background notifications only. The test
            # itself polls background_status, which is an ordinary IPC reply.
            if payload.get("type") in ("background_progress", "background_completed"):
                background_writes.append(payload["type"])
            if payload.get("type") == "background_progress":
                raise BrokenPipeError(32, "test client disconnected")
            return original_send(payload)

        with patch.object(client, "send", side_effect=send_with_disconnected_reader):
            self.manager.resume.set()
            # Resetting the fake KV precedes final job bookkeeping. Wait for
            # the terminal state, not merely for reset_count to change.
            deadline = time.monotonic() + 3
            while True:
                self.send({"type": "background_status",
                           "background_id": accepted["background_id"]})
                status = self.receive()
                self.assertEqual(status["type"], "background_status")
                if status["job"]["state"] == "CANCELLED":
                    break
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            self.assertEqual(self.manager.agent_chat.reset_count, 1)
            self.assertNotIn("step2", self.manager.events)
            self.assertEqual(self.manager.agent_chat.working_kv_tokens, 200)
            self.assertEqual(background_writes, ["background_progress"])
            snapshot = self.request({"type": "snapshot"})
            self.assertEqual(snapshot["type"], "snapshot")
            self.assertEqual(snapshot["status"]["background_jobs"], [])

    def test_explicit_cancel_marks_job_and_disables_followup(self):
        accepted = self.request({
            "type": "background_task", "agent": "agent_1",
            "steps": ["step1", "step2"],
        })
        self.assertTrue(self.manager.entered.wait(2))
        reply = self.request({
            "type": "cancel_background",
            "background_id": accepted["background_id"],
        })
        self.assertTrue(reply["cancelled"])
        self.manager.resume.set()
        for _ in range(100):
            if self.manager.agent_chat.reset_count:
                break
            time.sleep(.01)
        self.assertNotIn("step2", self.manager.events)
        self.assertEqual(self.manager.agent_chat.working_kv_tokens, 200)


if __name__ == "__main__":
    unittest.main()
