"""Stage 5B: autonomous QUERY result IPC, no live LiteRT model."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from agent_core_tools.server import CoreServer
from test_agent_core_tools_priority import FakeManager


class QueryResultIpcTests(unittest.TestCase):
    def test_disconnected_client_keeps_result_for_later_inbox_delivery(self):
        with tempfile.TemporaryDirectory() as directory:
            server = CoreServer(Path(directory) / "core.sock", manager=FakeManager())
            record = {
                "type": "task_query_result", "run_id": "R2",
                "task_id": 1, "source": "timer", "agent_id": "agent_1",
                "agent_ok": True, "text": "8.92°C",
            }
            try:
                server._publish_task_query_result(record)
                self.assertEqual(server._query_results["R2"], record)
            finally:
                server.close()

    def test_query_result_does_not_write_into_human_manager_dialogue(self):
        with tempfile.TemporaryDirectory() as d:
            manager = FakeManager()
            server = CoreServer(Path(d) / "core.sock", manager=manager)
            server.start()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.settimeout(3)
                    conn.connect(str(server.path))
                    with conn.makefile("r", encoding="utf-8") as rd:
                        def send(item):
                            conn.sendall((json.dumps(item, ensure_ascii=False) + "\n").encode())
                        def receive():
                            return json.loads(rd.readline())
                        send({"type": "acquire", "client": "owner"})
                        self.assertEqual(receive()["type"], "acquired")
                        record = {
                            "type": "task_query_result", "run_id": "R1",
                            "task_id": 5, "source": "timer",
                            "text": "Каталог /opt/model", "agent_id": "agent_1",
                            "agent_ok": True,
                        }
                        server._publish_task_query_result(record)
                        unsolicited = receive()
                        self.assertEqual(unsolicited, record)
                        send({"type": "task_query_results"})
                        reply = receive()
                        self.assertEqual(reply["type"], "task_query_results")
                        self.assertEqual(reply["items"], [record])
                        send({"type": "release"})
                        self.assertEqual(receive()["type"], "released")
                self.assertFalse(any(e.startswith("human:") or e.startswith("query:") for e in manager.events))
            finally:
                server.close()
                thread.join(2)
