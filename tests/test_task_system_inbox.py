"""Stage 5C: persistent, run-keyed QUERY inbox owned by Task SYSTEM."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import tempfile
import socket
import threading
import unittest
from unittest.mock import patch

from orchestration.tasks import TaskStoreError
from task_system.client import RemoteSystemRuntime, TaskRPC
from task_system.engine import TaskEngine
from task_system.main import Server, Handler


class RPC:
    def __init__(self, engine, peer="worker"):
        self.engine, self.peer = engine, peer

    def call(self, target, method, *args, **kwargs):
        value = self.engine.call(target, method, args, kwargs, peer=self.peer)
        return json.loads(json.dumps(
            value, default=lambda v: asdict(v) if is_dataclass(v) else v,
        ))


class QueryInboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.path = root / "state.db"
        self.task_file = root / "legacy-tasks.txt"
        self.event_file = root / "legacy-events.json"
        self.engine = self.open_engine()
        self.addCleanup(lambda: self.engine.close())
        self.rpc = RPC(self.engine)
        self.worker = RemoteSystemRuntime("litert", rpc=self.rpc)

    def open_engine(self):
        return TaskEngine(self.path, task_file=self.task_file,
                          event_file=self.event_file,
                          alive=lambda peer: peer == "worker")

    def start_run(self, method="query"):
        task = self.worker.create_periodic_task(
            "query", "read /opt/model", ("shell",), 3600,
            method=method, executor="litert",
        )
        self.rpc.call("system", "run_task", task.task_id)
        event, = self.worker.poll_due()
        self.worker.accepted(event)
        self.assertTrue(self.worker.begin_run(event))
        self.assertTrue(self.worker.valid_run(event))
        return event

    def notice(self, event, text="Каталог: /opt/model"):
        return {
            "type": "task_query_result", "run_id": event.run_id,
            "task_id": event.task_id, "source": event.source,
            "agent_id": "agent_1", "agent_ok": True, "text": text,
        }

    def publish(self, event, notice):
        return self.rpc.call(
            "worker", "publish_result", owner=self.worker.owner,
            run_id=event.run_id, offer_id=event.offer_id, result=notice,
        )

    def list_pending(self):
        return self.rpc.call("notifications", "list")

    def test_durable_after_restart_and_explicit_ack(self):
        event = self.start_run()
        payload = self.notice(event)
        self.assertTrue(self.publish(event, payload))
        self.worker.finish_run(event, "done")
        self.assertTrue(self.worker.flush_completions())
        self.assertEqual(self.list_pending(), [payload])
        self.engine.close()
        self.engine = self.open_engine()
        self.rpc.engine = self.engine
        self.assertEqual(self.list_pending(), [payload])
        self.assertTrue(self.rpc.call("notifications", "ack", event.run_id))
        self.assertTrue(self.rpc.call("notifications", "ack", event.run_id))
        self.assertEqual(self.list_pending(), [])
        self.engine.close()
        self.engine = self.open_engine()
        self.rpc.engine = self.engine
        self.assertEqual(self.list_pending(), [])

    def test_store_survives_service_restart_before_finish(self):
        event = self.start_run()
        payload = self.notice(event)
        self.assertTrue(self.publish(event, payload))
        self.assertEqual(self.engine.runs[event.run_id]["state"], "running")
        self.engine.close()
        self.engine = self.open_engine()
        self.rpc.engine = self.engine
        self.assertEqual(self.list_pending(), [payload])
        # The result was persisted before the original worker FINISH RPC.
        # A resumed finish is safe and does not execute tools.
        self.worker.finish_run(event, "done")
        self.assertTrue(self.worker.flush_completions())
        self.assertEqual(self.engine.runs[event.run_id]["state"], "done")
        self.assertEqual(self.list_pending(), [payload])

    def test_unix_rpc_can_read_and_ack_persisted_notification(self):
        event = self.start_run()
        payload = self.notice(event)
        self.assertTrue(self.publish(event, payload))
        path = self.path.with_name("inbox.sock")
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.close()
        except PermissionError:
            self.skipTest("Unix sockets unavailable")
        with Server(str(path), Handler) as server:
            server.engine = self.engine
            thread = threading.Thread(target=server.serve_forever,
                                      kwargs={"poll_interval": .02}, daemon=True)
            thread.start()
            try:
                rpc = TaskRPC(path)
                self.assertEqual(rpc.call("notifications", "list"), [payload])
                self.assertTrue(rpc.call("notifications", "ack", event.run_id))
                self.assertEqual(rpc.call("notifications", "list"), [])
            finally:
                server.shutdown()
                thread.join(2)

    def test_duplicate_publish_is_idempotent_even_after_finish(self):
        event = self.start_run()
        payload = self.notice(event)
        self.assertTrue(self.publish(event, payload))
        self.worker.finish_run(event, "done")
        self.assertTrue(self.worker.flush_completions())
        self.assertTrue(self.publish(event, payload))
        self.assertEqual(self.list_pending(), [payload])
        with self.assertRaisesRegex(TaskStoreError, "conflicting"):
            self.publish(event, self.notice(event, "ДРУГОЙ РЕЗУЛЬТАТ"))
        self.assertEqual(self.list_pending(), [payload])

    def test_foreign_owner_and_wrong_offer_cannot_publish(self):
        event = self.start_run()
        payload = self.notice(event)
        self.assertFalse(self.rpc.call(
            "worker", "publish_result", owner="other",
            run_id=event.run_id, offer_id=event.offer_id, result=payload,
        ))
        self.assertFalse(self.rpc.call(
            "worker", "publish_result", owner=self.worker.owner,
            run_id=event.run_id, offer_id="wrong", result=payload,
        ))
        self.assertEqual(self.list_pending(), [])
        self.assertTrue(self.publish(event, payload))

    def test_stopped_run_cannot_publish_and_no_notice_for_task_method(self):
        event = self.start_run()
        self.worker.stop_task(event.task_id)
        self.assertFalse(self.publish(event, self.notice(event)))
        self.assertEqual(self.list_pending(), [])

    def test_non_query_task_cannot_publish(self):
        event = self.start_run(method="task")
        self.assertFalse(self.publish(event, self.notice(event)))
        self.assertEqual(self.list_pending(), [])

    def test_sqlite_commit_failure_rolls_back_notification(self):
        event = self.start_run()
        payload = self.notice(event)
        with patch.object(self.engine, "_save", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.publish(event, payload)
        self.assertEqual(self.list_pending(), [])
        self.assertTrue(self.publish(event, payload))
        self.assertEqual(self.list_pending(), [payload])

    def test_payload_validation_and_unknown_ack(self):
        event = self.start_run()
        payload = self.notice(event)
        payload["text"] = "x" * 4097
        with self.assertRaisesRegex(TaskStoreError, "invalid QUERY"):
            self.publish(event, payload)
        self.assertFalse(self.rpc.call("notifications", "ack", "unknown"))
        with self.assertRaisesRegex(TaskStoreError, "invalid notification"):
            self.rpc.call("notifications", "list", limit=10000)
        self.assertEqual(self.list_pending(), [])


if __name__ == "__main__":
    unittest.main()
