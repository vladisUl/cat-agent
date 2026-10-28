"""Stage 6A opt-in native terminal: dialogue, durable QUERY and explicit ACK."""
from __future__ import annotations

import queue
import tempfile
from pathlib import Path
import threading
import time
import unittest

from agent_core_tools.native_terminal import NativeTerminalTUI
from agent_core_tools.server import CoreServer
from litert_agent.terminal import CoreClient
from test_agent_core_tools_priority import FakeManager


NOTICE = {
    "type": "task_query_result", "run_id": "test-run-one",
    "task_id": 1, "source": "manual", "agent_id": "agent_1",
    "agent_ok": True, "text": "Полный рабочий каталог: /opt/model",
}


class FakeClient:
    def __init__(self):
        self.incoming = queue.SimpleQueue()
        self.sent = []

    def send(self, message):
        self.sent.append(dict(message))


class NativeTerminalTests(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.ui = NativeTerminalTUI(self.client, {"model": "test"}, {"state": "IDLE"})

    def receive(self, message):
        self.client.incoming.put(message)
        self.ui._poll_core()

    def test_legacy_manager_reply_is_rendered_without_duplicate_completed(self):
        self.receive({"type": "accepted", "request_id": "r1"})
        self.receive({"type": "reply", "request_id": "r1", "text": "Привет"})
        self.receive({"type": "completed", "request_id": "r1", "text": "Привет"})
        self.assertEqual(list(self.ui._dialog)[-1][0:2], ("MANAGER", "Привет"))
        self.assertEqual(len(self.ui._dialog), 1)
        self.receive({"type": "completed", "request_id": "r1",
                      "chat_open": True})
        self.assertTrue(self.ui.status["chat_open"])
        self.receive({"type": "snapshot", "status": {"state": "IDLE"}})
        self.assertTrue(self.ui.status["chat_open"])
        self.receive({"type": "error", "kind": "error",
                      "text": "Native inference unavailable"})
        self.assertEqual(self.ui._dialog[-1][:2],
                         ("CORE", "Native inference unavailable"))

    def test_query_from_live_and_inbox_is_displayed_once_and_acked_after_insertion(self):
        self.receive(dict(NOTICE))
        self.receive({"type": "task_query_results", "items": [dict(NOTICE)]})
        self.assertEqual(len(self.ui._dialog), 1)
        self.assertEqual(self.ui._dialog[-1][:2], ("TASK QUERY", NOTICE["text"]))
        self.assertEqual(self.client.sent, [])  # No ACK before the display update.
        self.ui._request_snapshot()
        self.assertIn({"type": "task_query_results"}, self.client.sent)
        self.assertIn(
            {"type": "task_query_ack", "run_id": NOTICE["run_id"]}, self.client.sent,
        )
        self.receive({
            "type": "task_query_acked", "run_id": NOTICE["run_id"],
            "acknowledged": True,
        })
        self.assertEqual(self.ui._pending_query_ack, {})
        self.receive({"type": "task_query_results", "items": [dict(NOTICE)]})
        self.assertEqual(len(self.ui._dialog), 1)
        before = len(self.client.sent)
        self.ui._next_inbox_poll = time.monotonic() + 9999
        self.ui._request_snapshot()
        self.assertEqual(len(self.client.sent), before)

    def test_lost_ack_answer_retries_only_ack_and_not_dialogue(self):
        self.receive(dict(NOTICE))
        self.ui._request_snapshot()
        self.ui._pending_query_ack[NOTICE["run_id"]] = 0.0
        before = len(self.client.sent)
        self.ui._request_snapshot()
        self.assertEqual(
            self.client.sent[before:],
            [{"type": "task_query_ack", "run_id": NOTICE["run_id"]}],
        )
        self.assertEqual(len(self.ui._dialog), 1)

    def test_bad_query_data_does_not_ack_or_display(self):
        self.receive({"type": "task_query_result", "text": "missing run"})
        self.receive({"type": "task_query_results", "items": ["bad", None]})
        self.assertEqual(len(self.ui._dialog), 0)
        self.assertEqual(self.ui._pending_query_ack, {})

    def test_disconnect_stops_terminal(self):
        self.receive({"type": "disconnected", "text": "CORE stopped"})
        self.assertTrue(self.ui._quit)
        self.assertEqual(self.ui._dialog[-1][:2], ("CORE", "CORE stopped"))


class DurableInbox:
    def __init__(self):
        self.items = {NOTICE["run_id"]: dict(NOTICE)}
        self.acks = []
    def list_results(self):
        return list(self.items.values())
    def ack_result(self, run_id):
        self.acks.append(run_id)
        return self.items.pop(run_id, None) is not None
    def snapshot(self):
        return {"enabled": True}
    def close(self):
        pass


class NativeTerminalUnixTests(unittest.TestCase):
    def test_real_native_core_session_uses_manager_and_query_inbox(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manager = FakeManager()
            server = CoreServer(root / "native.sock", manager=manager)
            server.start()
            inbox = DurableInbox()
            server._task_worker = inbox
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            client = CoreClient(server.path)
            try:
                first = client.connect()
                self.assertEqual(first["type"], "acquired")
                ui = NativeTerminalTUI(client, first["core"], first["status"])

                client.send({"type": "user", "text": "Привет"})
                ui._request_snapshot()
                deadline = time.monotonic() + 3
                while not any(x[0] == "MANAGER" for x in ui._dialog):
                    self.assertLess(time.monotonic(), deadline)
                    ui._poll_core()
                    time.sleep(.01)
                self.assertTrue(any(
                    x[0] == "MANAGER" and "human says Привет" in x[1]
                    for x in ui._dialog
                ))

                # On startup an unread QUERY can be recovered from
                # TASK SYSTEM even if no live hint was received.
                deadline = time.monotonic() + 3
                while not any(x[0] == "TASK QUERY" for x in ui._dialog):
                    self.assertLess(time.monotonic(), deadline)
                    ui._poll_core()
                    time.sleep(.01)
                self.assertEqual(sum(x[0] == "TASK QUERY" for x in ui._dialog), 1)
                ui._request_snapshot()
                deadline = time.monotonic() + 3
                while inbox.items:
                    self.assertLess(time.monotonic(), deadline)
                    ui._poll_core()
                    time.sleep(.01)
                self.assertEqual(inbox.acks, [NOTICE["run_id"]])
                ui._poll_core()
                self.assertFalse(ui._pending_query_ack)
            finally:
                client.close()
                server.close()
                thread.join(2)
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
