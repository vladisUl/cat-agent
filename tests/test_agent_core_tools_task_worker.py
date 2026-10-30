"""Stage 5A integration tests: real TaskEngine grant policy, fake model.

Test the authoritative worker protocol, not a second private task registry.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from agent_core_tools.priority_scheduler import PriorityScheduler
from agent_core_tools.task_worker import NativeTaskWorker
from orchestration.tasks import TaskStoreError
from task_system.engine import TaskEngine
from task_system.client import TaskRPC
from task_system.main import Server, Handler
from test_agent_core_tools_priority import FakeManager


class LocalRPC:
    def __init__(self, engine, peer="native"):
        self.engine = engine
        self.peer = peer
        self.fail_finish_once = False
        self.fail_publish_once = False
        self.fail_publish_reply_once = False
        self.requests = []

    def call(self, target, operation, *args, **kwargs):
        self.requests.append((target, operation, kwargs))
        if target == "worker" and operation == "finish" and self.fail_finish_once:
            self.fail_finish_once = False
            raise TaskStoreError("test RPC timeout after inference")
        if target == "worker" and operation == "publish_result" and self.fail_publish_once:
            self.fail_publish_once = False
            raise TaskStoreError("test QUERY store unavailable before write")
        response = self.engine.call(target, operation, args, kwargs, peer=self.peer)
        if target == "worker" and operation == "publish_result" and self.fail_publish_reply_once:
            self.fail_publish_reply_once = False
            raise TaskStoreError("test QUERY store response lost AFTER commit")
        return json.loads(json.dumps(
            response,
            default=lambda obj: asdict(obj) if is_dataclass(obj) else obj,
        ))


class TaskWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.engine = TaskEngine(
            root / "tasks.sqlite3",
            task_file=root / "tasks.txt",
            event_file=root / "events.json",
            alive=lambda peer: peer == "native",
        )
        self.addCleanup(self.engine.close)
        self.rpc = LocalRPC(self.engine)
        self.manager = FakeManager()
        self.manager.run_agent = lambda ident, task: self.manager.pool.run(task, ident)
        self.scheduler = PriorityScheduler()
        self.addCleanup(self.scheduler.close)
        self.worker = NativeTaskWorker(
            self.scheduler, self.manager, rpc=self.rpc,
            agent_permissions={"agent_1": ("shell",)},
            poll_seconds=10,  # Explicit poll_once() in deterministic tests.
        )
        self.worker.start()
        self.addCleanup(self.worker.close)

    def make_task(self, skills=("shell",), method="task"):
        task = self.engine.runtime.create_periodic_task(
            "stage5-test", "TASK_TEST_ACTION", skills, 60,
            method=method, executor="litert",
        )
        timer = self.engine.runtime._task_timers[task.task_id]
        self.engine.tick(timer.next_fire_monotonic + 0.01)
        return task

    def wait_model_done(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.worker.snapshot()["unacknowledged"]:
                return
            time.sleep(0.005)
        self.fail("No model completion submitted for authoritative finish")

    def test_display_countdown_reads_authoritative_system_timer(self):
        task = self.engine.runtime.create_periodic_task(
            "status-timer", "READ", ("shell",), 60,
            method="query", executor="litert",
        )
        self.worker._refresh_task_display()
        snapshot = self.worker.snapshot()
        entry = next(t for t in snapshot["tasks"] if t["task_id"] == task.task_id)
        self.assertEqual(entry["executor"], "litert")
        self.assertEqual(entry["timer"]["period_seconds"], 60)
        self.assertGreater(entry["timer"]["next_fire_monotonic"], time.monotonic())
        self.assertIsNone(snapshot["task_display_error"])
        task_rpc_reads = [op for _, op, _ in self.rpc.requests
                          if op in ("task_snapshot", "task_timer_snapshot")]
        self.assertEqual(task_rpc_reads[-2:], ["task_snapshot", "task_timer_snapshot"])
        self.worker._refresh_task_display()
        self.assertEqual(len([op for _, op, _ in self.rpc.requests
                              if op in ("task_snapshot", "task_timer_snapshot")]),
                         len(task_rpc_reads))
        self.engine.runtime.stop_task(task.task_id)
        self.worker._next_display_refresh = 0.0
        self.worker._refresh_task_display()
        item = next(t for t in self.worker.snapshot()["tasks"]
                    if t["task_id"] == task.task_id)
        self.assertFalse(item["enabled"])
        self.assertFalse(item["timer"]["enabled"])

    def test_task_run_exactly_once_and_durable_finish(self):
        task = self.make_task()
        self.assertTrue(self.worker.poll_once())
        self.wait_model_done()
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)
        self.assertFalse(self.worker.poll_once())  # Flush completion; no new offer.
        record, = list(self.engine.runs.values())
        self.assertEqual(record["state"], "done")
        self.assertEqual(record["event"]["task_id"], task.task_id)
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)
        self.assertEqual(self.manager.agent_chat.working_kv_tokens, 200)
        operations = [op for _, op, _ in self.rpc.requests]
        self.assertIn("heartbeat", operations)
        self.assertIn("pull", operations)
        self.assertIn("begin", operations)
        self.assertIn("valid", operations)
        self.assertIn("finish", operations)

    def test_finish_rpc_outage_retries_ack_only(self):
        self.rpc.fail_finish_once = True
        self.make_task()
        self.assertTrue(self.worker.poll_once())
        self.wait_model_done()
        self.assertFalse(self.worker.poll_once())  # finish RPC failed
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 1)
        self.assertFalse(self.worker.poll_once())  # finish RPC recovered
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 0)
        record, = list(self.engine.runs.values())
        self.assertEqual(record["state"], "done")
        self.assertEqual(len([r for r in self.rpc.requests if r[1]=="begin"]), 1)

    def test_stopped_task_never_executes_after_offer(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        self.scheduler.submit("background", lambda: (entered.set(), release.wait(3)))
        self.assertTrue(entered.wait(2))
        task = self.make_task()
        self.assertTrue(self.worker.poll_once())
        self.engine.runtime.stop_task(task.task_id)
        release.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not self.worker.snapshot()["reserved_agents"]:
                break
            time.sleep(.005)
        self.assertEqual(self.worker.snapshot()["reserved_agents"], [])
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 0)
        record, = list(self.engine.runs.values())
        self.assertEqual(record["state"], "cancelled")

    def test_agent_allowlist_requires_true_subset_of_task_skills(self):
        cases = [
            (("mqtt",), ("shell",), False),            # disjoint sets
            (("shell", "mqtt"), ("shell",), False), # missing a required tool
            (("shell",), ("shell",), True),         # exact match
            (("shell",), ("shell", "mqtt"), True), # allowed superset
            ((), (), True),                         # no tool permissions needed
        ]
        for required, granted, expected in cases:
            with self.subTest(required=required, granted=granted):
                self.worker.permissions["agent_1"] = frozenset(granted)
                selected = self.worker._select_agent({"skills": list(required)})
                self.assertEqual(selected == "agent_1", expected)
        self.assertEqual(self.manager.events, [])

    def test_target_agent_is_selected_without_tool_names_in_task(self):
        self.worker.permissions["agent_2"] = frozenset({"shell"})
        original_status = self.manager.agent_status
        try:
            self.manager.agent_status = lambda: [
                {"agent_id": "agent_1", "state": "FREE"},
                {"agent_id": "agent_2", "state": "FREE"},
            ]
            self.assertEqual(
                self.worker._select_agent({"agent_id": "agent_2", "skills": []}),
                "agent_2",
            )
            self.assertEqual(
                self.worker._select_agent({"agent_id": "agent_1", "skills": []}),
                "agent_1",
            )
            self.assertIsNone(
                self.worker._select_agent({"agent_id": "agent_3", "skills": []}),
            )
            self.manager.agent_status = lambda: [
                {"agent_id": "agent_1", "state": "FREE"},
                {"agent_id": "agent_2", "state": "RUNNING"},
            ]
            self.assertIsNone(
                self.worker._select_agent({"agent_id": "agent_2", "skills": []}),
            )
        finally:
            self.manager.agent_status = original_status

    def test_unmatched_skill_is_deferred_without_begin_and_next_run_executes(self):
        denied = self.make_task(skills=("mqtt",))
        denied_run = next(iter(self.engine.runs))
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.worker.snapshot()["offered"], 0)
        self.assertEqual(self.manager.events, [])
        self.assertFalse(any(
            target == "worker" and method in ("begin", "valid")
            for target, method, _ in self.rpc.requests
        ))
        self.assertEqual(self.engine.runs[denied_run]["state"], "pending")
        allowed = self.make_task(skills=("shell",))
        self.assertNotEqual(denied.task_id, allowed.task_id)
        self.assertTrue(self.worker.poll_once())
        self.wait_model_done()
        self.assertFalse(self.worker.poll_once())
        runs = list(self.engine.runs.values())
        self.assertEqual(runs[0]["state"], "pending")
        self.assertEqual(runs[1]["state"], "done")
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)

    def test_query_agent_then_separate_manager_and_result_publisher(self):
        from orchestration_tools.contracts import ManagerTurn
        summaries = []
        self.worker.on_query_result = summaries.append
        def report(task_id, result, *, failed=False):
            self.manager.events.append("query-manager")
            self.assertEqual(task_id, 1)
            self.assertIn("TASK_TEST_ACTION", result)
            return ManagerTurn("reply", "TASK QUERY summary: " + result)
        self.manager.process_task_query = report
        self.make_task(method="query")
        self.assertTrue(self.worker.poll_once())
        self.wait_model_done()
        self.assertEqual(summaries, [])  # No IPC hint before durable commit.
        self.assertFalse(self.worker.poll_once())  # STORE, notify, FINISH.
        self.assertEqual(len(summaries), 1)
        self.assertEqual(self.manager.events, ["TASK_TEST_ACTION", "query-manager"])
        self.assertEqual(summaries[0]["task_id"], 1)
        self.assertIn("TASK QUERY summary", summaries[0]["text"])
        self.assertEqual(self.manager.agent_chat.working_kv_tokens, 200)
        self.assertFalse(self.worker.poll_once())  # Flush finish, no new RUN.
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "done")

    def test_query_store_outage_does_not_publish_or_rerun_model(self):
        from orchestration_tools.contracts import ManagerTurn
        events = []
        self.worker.on_query_result = events.append
        self.manager.process_task_query = (
            lambda _id, _text, **_kw: ManagerTurn("reply", "Сводка QUERY")
        )
        self.rpc.fail_publish_once = True
        self.make_task(method="query")
        self.assertTrue(self.worker.poll_once())
        self.wait_model_done()
        self.assertFalse(self.worker.poll_once())  # Failed to persist.
        self.assertEqual(events, [])
        self.assertEqual(self.worker.snapshot()["pending_query_stores"], 1)
        self.assertEqual(self.rpc.call("notifications", "list"), [])
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)
        self.assertFalse(self.worker.poll_once())  # Exactly one retry of STORE.
        self.assertEqual(len(events), 1)
        self.assertEqual(len(self.rpc.call("notifications", "list")), 1)
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "done")

    def test_lost_query_store_reply_is_idempotent(self):
        from orchestration_tools.contracts import ManagerTurn
        notifications = []
        self.worker.on_query_result = notifications.append
        self.manager.process_task_query = (
            lambda _id, _text, **_kw: ManagerTurn("reply", "Сводка QUERY")
        )
        self.rpc.fail_publish_reply_once = True
        self.make_task(method="query")
        self.assertTrue(self.worker.poll_once())
        self.wait_model_done()
        self.assertFalse(self.worker.poll_once())  # Persisted, reply lost.
        self.assertEqual(notifications, [])
        self.assertEqual(len(self.rpc.call("notifications", "list")), 1)
        self.assertFalse(self.worker.poll_once())  # Same RUN, same immutable data.
        self.assertEqual(len(notifications), 1)
        self.assertEqual(len(self.rpc.call("notifications", "list")), 1)
        self.assertEqual(self.manager.events.count("TASK_TEST_ACTION"), 1)
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "done")

    def test_query_yields_to_human_between_agent_and_manager(self):
        from orchestration_tools.contracts import ManagerTurn
        from unittest.mock import patch
        self.manager.process_task_query = lambda *args, **kwargs: (
            self.manager.events.append("query-manager")
            or ManagerTurn("reply", "Summary")
        )
        self.make_task(method="query")
        original_submit = self.scheduler.submit
        def submit(kind, callback, *, continuation=False):
            if continuation:
                original_submit("human", lambda: self.manager.events.append("human"))
            return original_submit(kind, callback, continuation=continuation)
        with patch.object(self.scheduler, "submit", side_effect=submit):
            self.assertTrue(self.worker.poll_once())
            deadline = time.monotonic() + 3
            while self.worker.snapshot()["unacknowledged"] == 0 and time.monotonic() < deadline:
                time.sleep(.005)
        self.assertEqual(self.manager.events[:3],
                         ["TASK_TEST_ACTION", "human", "query-manager"])

    def test_deferred_grant_uses_correct_offer_and_can_go_to_other_worker(self):
        denied = self.make_task(skills=("mqtt",))
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "pending")
        # Different litert worker has the required skill; native worker's
        # local skip is not a server-wide blacklist.
        from task_system.client import RemoteSystemRuntime
        other = RemoteSystemRuntime("litert", rpc=LocalRPC(self.engine, peer="other"))
        self.engine.alive = lambda peer: peer in ("native", "other")
        offer, = other.poll_due()
        self.assertEqual(offer.task_id, denied.task_id)
        self.assertTrue(other.begin_run(offer))
        other.finish_run(offer, "done")
        self.assertTrue(other.flush_completions())
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "done")

    def test_human_priority_over_waiting_task(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        order = self.manager.events
        self.scheduler.submit(
            "background", lambda: (order.append("blocking"),
                                   entered.set(), release.wait(3)),
        )
        self.assertTrue(entered.wait(2))
        self.make_task()
        self.assertTrue(self.worker.poll_once())
        self.scheduler.submit("human", lambda: order.append("human"))
        release.set()
        self.wait_model_done()
        self.assertEqual(order[:3], ["blocking", "human", "TASK_TEST_ACTION"])

    def test_revocation_between_agent_and_query_manager_blocks_report(self):
        from orchestration_tools.contracts import ManagerTurn
        from unittest.mock import patch
        task = self.make_task(method="query")
        self.manager.process_task_query = lambda *_args, **_kw: (
            self.manager.events.append("query-manager")
            or ManagerTurn("reply", "Summary")
        )
        original_submit = self.scheduler.submit
        def submit(kind, callback, *, continuation=False):
            if continuation:
                original_submit("human", lambda: self.engine.runtime.stop_task(task.task_id))
            return original_submit(kind, callback, continuation=continuation)
        with patch.object(self.scheduler, "submit", side_effect=submit):
            self.assertTrue(self.worker.poll_once())
            self.wait_model_done()
        self.assertEqual(self.manager.events, ["TASK_TEST_ACTION"])
        self.worker.poll_once()
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "cancelled")

    def test_begin_failure_never_invokes_model(self):
        self.make_task()
        self.rpc.fail_begin_once = True
        original = self.rpc.call
        def call(target, operation, *args, **kwargs):
            if target == "worker" and operation == "begin" and self.rpc.fail_begin_once:
                self.rpc.fail_begin_once = False
                raise TaskStoreError("begin unavailable")
            return original(target, operation, *args, **kwargs)
        self.rpc.call = call
        self.assertTrue(self.worker.poll_once())
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if not self.worker.snapshot()["reserved_agents"]:
                break
            time.sleep(.005)
        self.assertEqual(self.manager.events, [])
        # Offer is allowed to return to the next pull, since no model ran.
        self.assertEqual(self.worker.snapshot()["reserved_agents"], [])
        # An unanswered BEGIN may have committed server-side. The uncertain
        # grant is held for a FINISH-only cleanup, not treated as never seen.
        self.assertEqual(self.worker.snapshot()["unacknowledged"], 1)
        self.assertFalse(self.worker.poll_once())
        self.assertEqual(self.worker.snapshot()["offered"], 0)
        self.assertEqual(next(iter(self.engine.runs.values()))["state"], "pending")


class TaskWorkerUnixTests(unittest.TestCase):
    def test_real_unix_rpc_claims_and_finishes_one_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = TaskEngine(
                root / "state.db",
                task_file=root / "tasks.txt",
                event_file=root / "events.json",
            )
            try:
                with Server(str(root / "tasks.sock"), Handler) as service:
                    service.engine = engine
                    service_thread = threading.Thread(
                        target=service.serve_forever,
                        kwargs={"poll_interval": .02}, daemon=True,
                    )
                    service_thread.start()
                    try:
                        manager = FakeManager()
                        manager.run_agent = (
                            lambda ident, task: manager.pool.run(task, ident)
                        )
                        scheduler = PriorityScheduler()
                        worker = NativeTaskWorker(
                            scheduler, manager,
                            rpc=TaskRPC(root / "tasks.sock"),
                            agent_permissions={"agent_1": ("shell",)},
                            poll_seconds=10,
                        )
                        try:
                            worker.start()
                            task = engine.runtime.create_periodic_task(
                                "real-socket", "TASK_UNIX_ACTION",
                                ("shell",), 60, method="task", executor="litert",
                            )
                            rpc = TaskRPC(root / "tasks.sock")
                            rpc.call("system", "run_task", task.task_id)
                            self.assertTrue(worker.poll_once())
                            deadline = time.monotonic() + 3
                            while (not worker.snapshot()["unacknowledged"]
                                   and time.monotonic() < deadline):
                                time.sleep(.01)
                            self.assertEqual(
                                manager.events.count("TASK_UNIX_ACTION"), 1,
                            )
                            worker.poll_once()
                            runs = rpc.call("system", "runs")
                            self.assertEqual(runs[0]["state"], "done")
                            self.assertEqual(worker.snapshot()["unacknowledged"], 0)
                        finally:
                            scheduler.close()
                            worker.close()
                    finally:
                        service.shutdown()
                        service_thread.join(timeout=2)
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
