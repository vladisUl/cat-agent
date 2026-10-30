"""Stage 5D: fault-injection at authority and non-replay boundaries."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_core_tools.priority_scheduler import PriorityScheduler
from agent_core_tools.task_worker import NativeTaskWorker
from orchestration.tasks import TaskStoreError
from task_system.engine import TaskEngine
from test_agent_core_tools_priority import FakeManager


class FaultRpc:
    def __init__(self, engine):
        self.engine = engine
        self.fail_before = {}
        self.fail_after = {}
        self.requests = []

    def call(self, target, operation, *args, **kwargs):
        self.requests.append((target, operation, kwargs))
        key = (target, operation)
        if self.fail_before.get(key, 0):
            self.fail_before[key] -= 1
            raise TaskStoreError("injected network outage before commit")
        response = self.engine.call(target, operation, args, kwargs, peer="native")
        if self.fail_after.get(key, 0):
            self.fail_after[key] -= 1
            raise TaskStoreError("injected response lost after commit")
        return json.loads(json.dumps(
            response, default=lambda value: asdict(value) if is_dataclass(value) else value,
        ))


class WorkerFaultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.engine = TaskEngine(
            root / "task.sqlite3", task_file=root / "tasks.txt",
            event_file=root / "events.json", alive=lambda peer: peer == "native",
        )
        self.addCleanup(self.engine.close)
        self.rpc = FaultRpc(self.engine)
        self.manager = FakeManager()
        self.manager.run_agent = lambda ident, task: self.manager.pool.run(task, ident)
        self.scheduler = PriorityScheduler()
        self.addCleanup(self.scheduler.close)
        self.worker = NativeTaskWorker(
            self.scheduler, self.manager, rpc=self.rpc,
            agent_permissions={"agent_1": ("shell",)}, poll_seconds=10,
        )
        self.worker.start()
        self.addCleanup(self.worker.close)

    def task(self, method="task"):
        task = self.engine.runtime.create_periodic_task(
            "stage5d", "ACTION_STAGE5D", ("shell",), 3600,
            method=method, executor="litert",
        )
        self.rpc.call("system", "run_task", task.task_id)
        return task

    def run_record(self):
        return next(iter(self.engine.runs.values()))

    def await_state(self, predicate):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.005)
        self.fail("native stage 5D state transition timed out")

    def test_begin_committed_but_reply_lost_never_runs_model_or_sticks_running(self):
        self.task()
        self.rpc.fail_after[("worker", "begin")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertEqual(self.run_record()["state"], "running")
        self.assertEqual(self.manager.events, [])
        self.assertFalse(self.worker.poll_once())  # FINISH only.
        self.assertEqual(self.run_record()["state"], "interrupted")
        self.assertEqual(self.manager.events, [])
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)

    def test_begin_unavailable_before_commit_reoffers_unstarted_run(self):
        self.task()
        self.rpc.fail_before[("worker", "begin")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertEqual(self.run_record()["state"], "offered")
        self.assertEqual(self.manager.events, [])
        self.assertFalse(self.worker.poll_once())  # finish(offered) -> pending
        self.assertEqual(self.run_record()["state"], "pending")
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)
        self.assertEqual(self.manager.events, [])

    def test_valid_reply_lost_before_agent_never_runs_model(self):
        self.task()
        self.rpc.fail_after[("worker", "valid")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertEqual(self.manager.events, [])
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.run_record()["state"], "interrupted")

    def test_query_publish_reply_loss_retries_store_not_agent(self):
        task = self.task(method="query")
        self.rpc.fail_after[("worker", "publish_result")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertFalse(self.worker.poll_once())  # Store committed; reply lost.
        items = self.rpc.call("notifications", "list")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"], "ACTION_STAGE5D")
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.assertFalse(self.worker.poll_once())  # Retry same immutable store.
        self.assertEqual(self.run_record()["state"], "done")
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.assertEqual(len(self.rpc.call("notifications", "list")), 1)

    def test_query_store_unavailable_then_shutdown_does_not_rerun_agent(self):
        self.task(method="query")
        self.rpc.fail_before[("worker", "publish_result")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.rpc.call("notifications", "list"), [])
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.worker.close()  # Flush pending store/finish without any model call.
        self.assertEqual(self.run_record()["state"], "done")
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.assertEqual(len(self.rpc.call("notifications", "list")), 1)

    def test_shutdown_before_begin_returns_unstarted_offer_to_pending(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.scheduler.submit(
            "human", lambda: (entered.set(), release.wait(3)),
        )
        self.assertTrue(entered.wait(2))
        self.task()
        self.assertTrue(self.worker.poll_once())
        self.assertEqual(self.run_record()["state"], "offered")
        self.assertEqual(self.manager.events, [])
        closer = threading.Thread(target=self.scheduler.close)
        closer.start()
        self.await_state(lambda: self.scheduler._stopping)
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.worker.close()  # OFFLINE relinquishes unstarted offer.
        self.assertEqual(self.run_record()["state"], "pending")
        self.assertEqual(self.manager.events, [])
        self.assertEqual(self.rpc.call("notifications", "list"), [])

    def test_rpc_outage_during_pull_does_not_admit_work_twice(self):
        self.task()
        self.rpc.fail_before[("worker", "pull")] = 1
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.run_record()["state"], "pending")
        self.assertEqual(self.manager.events, [])
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.run_record()["state"], "done")
        self.assertEqual(self.manager.events.count("ACTION_STAGE5D"), 1)


if __name__ == "__main__":
    unittest.main()
