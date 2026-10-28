"""Stage 5C IPC: list durable TASK inbox and ACK by RUN, owner only."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from agent_core_tools.server import CoreServer
from task_system.engine import TaskEngine
from test_agent_core_tools_priority import FakeManager


class TaskRpc:
    def __init__(self, engine):
        self.engine = engine

    def call(self, target, method, *args, **kwargs):
        result = self.engine.call(target, method, args, kwargs, peer="test-peer")
        return json.loads(json.dumps(
            result, default=lambda obj: asdict(obj) if is_dataclass(obj) else obj,
        ))


class InboxAdapter:
    def __init__(self, rpc):
        self.rpc = rpc
        self.closed = False
    def list_results(self):
        return self.rpc.call("notifications", "list")
    def ack_result(self, run_id):
        return self.rpc.call("notifications", "ack", run_id)
    def snapshot(self):
        return {"enabled": True}
    def close(self):
        self.closed = True


class QueryInboxIpcTests(unittest.TestCase):
    def test_real_sqlite_inbox_survives_core_cache_and_explicit_ack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = TaskEngine(
                root / "tasks.db", task_file=root / "legacy-tasks.txt",
                event_file=root / "legacy-events.json",
                alive=lambda peer: peer == "test-peer",
            )
            try:
                rpc = TaskRpc(engine)
                task = rpc.call(
                    "system", "create_periodic_task",
                    "QUERY", "проверь каталог", ["shell"], 3600,
                    method="query", executor="litert",
                )
                rpc.call("system", "run_task", task["task_id"])
                offer = rpc.call("worker", "pull",
                                 owner="worker", executor="litert", seen=[])
                self.assertTrue(rpc.call(
                    "worker", "begin", owner="worker", run_id=offer["run_id"],
                    offer_id=offer["offer_id"],
                ))
                record = {
                    "type": "task_query_result",
                    "run_id": offer["run_id"], "task_id": task["task_id"],
                    "source": "manual", "agent_id": "agent_1",
                    "agent_ok": True, "text": "Каталог /opt/model",
                }
                self.assertTrue(rpc.call(
                    "worker", "publish_result", owner="worker",
                    run_id=offer["run_id"], offer_id=offer["offer_id"],
                    result=record,
                ))
                self.assertTrue(rpc.call(
                    "worker", "finish", owner="worker",
                    run_id=offer["run_id"], offer_id=offer["offer_id"],
                    outcome="done",
                ))

                server = CoreServer(root / "core.sock", manager=FakeManager())
                server.start()
                inbox = InboxAdapter(rpc)
                server._task_worker = inbox
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                        conn.settimeout(3)
                        conn.connect(str(server.path))
                        with conn.makefile("r", encoding="utf-8") as reader:
                            def send(data):
                                conn.sendall((json.dumps(data) + "\n").encode())
                                return json.loads(reader.readline())
                            self.assertEqual(send({
                                "type": "task_query_results",
                            })["type"], "busy")
                            self.assertEqual(send({
                                "type": "task_query_ack", "run_id": offer["run_id"],
                            })["type"], "busy")
                            self.assertEqual(send({
                                "type": "acquire", "client": "owner",
                            })["type"], "acquired")
                            self.assertEqual(send({
                                "type": "task_query_results",
                            })["items"], [record])
                            self.assertEqual(send({
                                "type": "task_query_ack", "run_id": offer["run_id"],
                            }), {
                                "type": "task_query_acked",
                                "run_id": offer["run_id"], "acknowledged": True,
                            })
                            self.assertEqual(send({
                                "type": "task_query_results",
                            })["items"], [])
                            self.assertEqual(send({"type": "release"})["type"], "released")
                    self.assertEqual(rpc.call("notifications", "list"), [])
                finally:
                    server.close()
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
                    self.assertTrue(inbox.closed)
            finally:
                engine.close()
