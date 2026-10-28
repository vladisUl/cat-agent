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
from orchestration_tools.contracts import ManagerTurn
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
        self.manager.process_task_query = self.query_manager
        self.scheduler = PriorityScheduler()
        self.addCleanup(self.scheduler.close)
        self.worker = NativeTaskWorker(
            self.scheduler, self.manager, rpc=self.rpc,
            agent_permissions={"agent_1": ("shell",)}, poll_seconds=10,
        )
        self.worker.start()
        self.addCleanup(self.worker.close)

    def query_manager(self, task_id, result, *, failed=False):
        self.manager.events.append("QUERY_MANAGER")
        return ManagerTurn("reply", "Report: " + result)

    def task(self, method="task"):
        task = self.engine.runtime.create_periodic_task(
            "stage5d", "ACTION_STAGE5D", ("shell",), 3600,
            method=method, executor="litert",
        )
        self.rpc.call("system", "run_task", task.task_id)
        return task

    def run(self):
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
        self.assertEqual(self.run()["state"], "running")
        self.assertEqual(self.manager.events, [])
        self.assertFalse(self.worker.poll_once())  # FINISH only.
        self.assertEqual(self.run()["state"], "interrupted")
        self.assertEqual(self.manager.events, [])
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)

    def test_begin_unavailable_before_commit_reoffers_unstarted_run(self):
        self.task()
        self.rpc.fail_before[("worker", "begin")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertEqual(self.run()["state"], "offered")
        self.assertEqual(self.manager.events, [])
        self.assertFalse(self.worker.poll_once())  # finish(offered) -> pending
        self.assertEqual(self.run()["state"], "pending")
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)
        self.assertEqual(self.manager.events, [])

    def test_valid_reply_lost_before_agent_never_runs_model(self):
        self.task()
        self.rpc.fail_after[("worker", "valid")] = 1
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertEqual(self.manager.events, [])
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.run()["state"], "interrupted")

    def test_query_waits_for_rpc_and_only_retries_manager_boundary(self):
        self.task(method="query")
        # First validity check is before AGENT; second is before QUERY manager.
        original_call = self.rpc.call
        seen_valid = 0
        def drop_second_valid(target, operation, *args, **kwargs):
            nonlocal seen_valid
            if (target, operation) == ("worker", "valid"):
                seen_valid += 1
                if seen_valid == 2:
                    raise TaskStoreError("Task SYSTEM temporarily disconnected")
            return original_call(target, operation, *args, **kwargs)
        with patch.object(self.rpc, "call", side_effect=drop_second_valid):
            self.assertTrue(self.worker.poll_once())
            self.await_state(lambda: self.worker.snapshot()["pending_query_validations"] == 1)
            self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
            self.assertEqual(self.worker.snapshot()["query_continuations"], 1)
            self.assertEqual(self.run()["state"], "running")
            self.assertFalse(self.worker.poll_once())  # retries validation
            self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
            self.assertEqual(self.manager.events, ["ACTION_STAGE5D", "QUERY_MANAGER"])
            self.assertFalse(self.worker.poll_once())  # persist and finish
        self.assertEqual(self.run()["state"], "done")
        self.assertEqual(self.manager.events.count("ACTION_STAGE5D"), 1)
        self.assertEqual(self.manager.events.count("QUERY_MANAGER"), 1)
        items = self.rpc.call("notifications", "list")
        self.assertEqual(len(items), 1)
        self.assertIn("Report:", items[0]["text"])

    def test_query_revoked_during_outage_never_invokes_manager(self):
        task = self.task(method="query")
        original_call = self.rpc.call
        seen_valid = 0
        def drop_second_valid(target, operation, *args, **kwargs):
            nonlocal seen_valid
            if (target, operation) == ("worker", "valid"):
                seen_valid += 1
                if seen_valid == 2:
                    raise TaskStoreError("temporary disconnect")
            return original_call(target, operation, *args, **kwargs)
        with patch.object(self.rpc, "call", side_effect=drop_second_valid):
            self.assertTrue(self.worker.poll_once())
            self.await_state(lambda: self.worker.snapshot()["pending_query_validations"] == 1)
            self.engine.runtime.stop_task(task.task_id)
            self.assertFalse(self.worker.poll_once())  # retry valid -> revoked
            self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
            self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.assertEqual(self.run()["state"], "cancelled")
        self.assertEqual(self.rpc.call("notifications", "list"), [])

    def test_shutdown_discards_queued_query_without_replaying_agent(self):
        self.task(method="query")
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        original_submit = self.scheduler.submit
        def queue_blocker(kind, callback, *, continuation=False):
            if continuation:
                original_submit("human", lambda: (entered.set(), release.wait(3)))
            return original_submit(kind, callback, continuation=continuation)
        with patch.object(self.scheduler, "submit", side_effect=queue_blocker):
            self.assertTrue(self.worker.poll_once())
            self.assertTrue(entered.wait(2))
        # The QUERY continuation is still queued behind the active human.
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.assertEqual(self.worker.snapshot()["query_continuations"], 1)
        closer = threading.Thread(target=self.scheduler.close)
        closer.start()
        self.await_state(lambda: self.scheduler._stopping)
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.worker.close()  # No manager execution; FINISH(interrupted).
        self.assertEqual(self.manager.events, ["ACTION_STAGE5D"])
        self.assertEqual(self.run()["state"], "interrupted")
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)
        self.assertEqual(self.worker.snapshot()["query_continuations"], 0)
        self.assertEqual(self.rpc.call("notifications", "list"), [])

    def test_shutdown_before_begin_returns_unstarted_offer_to_pending(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.scheduler.submit(
            "human", lambda: (entered.set(), release.wait(3)),
        )
        self.assertTrue(entered.wait(2))
        self.task()
        self.assertTrue(self.worker.poll_once())
        self.assertEqual(self.run()["state"], "offered")
        self.assertEqual(self.manager.events, [])
        closer = threading.Thread(target=self.scheduler.close)
        closer.start()
        self.await_state(lambda: self.scheduler._stopping)
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        self.worker.close()  # OFFLINE relinquishes unstarted offer.
        self.assertEqual(self.run()["state"], "pending")
        self.assertEqual(self.manager.events, [])
        self.assertEqual(self.rpc.call("notifications", "list"), [])

    def test_rpc_outage_during_pull_does_not_admit_work_twice(self):
        self.task()
        self.rpc.fail_before[("worker", "pull")] = 1
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.run()["state"], "pending")
        self.assertEqual(self.manager.events, [])
        self.assertTrue(self.worker.poll_once())
        self.await_state(lambda: self.worker.snapshot()["unacknowledged"] == 1)
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.run()["state"], "done")
        self.assertEqual(self.manager.events.count("ACTION_STAGE5D"), 1)


if __name__ == "__main__":
    unittest.main()
