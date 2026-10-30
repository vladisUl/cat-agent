"""Periodic native MANAGER tools: durable RPC, independent timer, safe boundaries."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import tempfile
import time
import unittest

from agent_core_tools.priority_scheduler import PriorityScheduler
from agent_core_tools.task_worker import NativeTaskWorker
from orchestration.tasks import TaskStoreError
from orchestration_tools.agents import AgentOutcome, AgentState
from orchestration_tools.contracts import ManagerTurn
from orchestration_tools.task_tools import NativeTaskController, create_task_tools
from task_system.engine import TaskEngine


class LocalRPC:
    def __init__(self, engine):
        self.engine = engine
        self.calls = []

    def call(self, target, operation, *args, **kwargs):
        self.calls.append((target, operation, args, kwargs))
        response = self.engine.call(target, operation, args, kwargs, peer="native")
        return json.loads(json.dumps(
            response,
            default=lambda value: asdict(value) if is_dataclass(value) else value,
        ))


class FakeAgentManager:
    def __init__(self):
        self.calls = []

    def agent_status(self):
        return [{"agent_id": "agent_1", "state": "FREE"}]

    def run_agent(self, agent_id, text):
        self.calls.append((agent_id, text))
        return AgentOutcome(agent_id, AgentState.FREE, "8.36°C", 250)

    def process_task_query(self, task_id, text, failed=False):
        return ManagerTurn("reply", f"TASK {task_id}: {text}")


class NativePeriodicTaskTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.engine = TaskEngine(
            root / "task.sqlite3", task_file=root / "legacy-task.txt",
            event_file=root / "legacy-events.json", alive=lambda peer: peer == "native",
        )
        self.addCleanup(self.engine.close)
        self.rpc = LocalRPC(self.engine)
        self.controller = NativeTaskController(
            self.rpc, {"agent_1": ("get_outdoor_weather", "shell")},
        )
        self.tools = {t.__name__: t for t in create_task_tools(self.controller)}

    def test_persistent_query_create_list_stop_and_deduplicate(self):
        create = self.tools["create_periodic_query"]
        self.assertFalse(create(60, "get_outdoor_weather", "измерь температуру")["ok"])
        self.assertEqual(self.rpc.calls, [])  # Warm-up must be side-effect free.
        self.controller.enabled = True
        self.assertFalse(create(60, "get_outdoor_weather", "измерь температуру")["ok"])
        self.controller.begin_turn()
        try:
            invalid = create(60, "arbitrary_tool", "постучи")
            self.assertFalse(invalid["ok"])
            self.assertEqual(self.rpc.calls, [])
            item = create(60, "get_outdoor_weather", "измерь температуру на улице")
            self.assertTrue(item["ok"], item)
            self.assertEqual(item["status"], "scheduled")
            self.assertEqual(item["delivery"], "TASK QUERY notifications inbox")
            self.assertEqual(item["period_seconds"], 60)
            self.assertEqual(len([c for c in self.rpc.calls if c[1] == "create_periodic_task"]), 1)
            again = create(60, "get_outdoor_weather", "измерь температуру на улице")
            self.assertFalse(again["ok"])
            self.assertEqual(len([c for c in self.rpc.calls if c[1] == "create_periodic_task"]), 1)
            records = self.tools["list_periodic_queries"]()
            self.assertEqual(len(records["tasks"]), 1)
            self.assertEqual(records["tasks"][0]["task_id"], item["task_id"])
            record = self.rpc.call("tasks", "require", item["task_id"])
            self.assertEqual(record["method"], "query")
            self.assertEqual(record["executor"], "litert")
            self.assertEqual(record["skills"], ["get_outdoor_weather"])
            self.assertEqual(record["timer_period_seconds"], 60.0)
            self.assertIn("Выполни ОДИН раз", record["text"])
            self.assertIn("get_outdoor_weather", record["text"])
            stopped = self.tools["stop_periodic_query"](item["task_id"])
            self.assertEqual(stopped, {
                "ok": True, "task_id": item["task_id"], "enabled": False,
            })
        finally:
            self.controller.finish_turn()
        self.assertFalse(self.tools["list_periodic_queries"]()["ok"])
        self.assertEqual(
            self.rpc.call("tasks", "require", item["task_id"])["enabled"], False,
        )
        # Timers belong to SYSTEM and remain persisted, including disabled state.
        self.assertTrue(any(t.task_id == item["task_id"] for t in
                            self.engine.runtime.task_timer_snapshot()))

    def test_periodic_query_executes_repeated_agent_turns_and_delivers_notifications(self):
        self.controller.enabled = True
        self.controller.begin_turn()
        try:
            item = self.controller.create(60, "get_outdoor_weather", "Температура на улице")
        finally:
            self.controller.finish_turn()
        self.assertTrue(item["ok"], item)
        manager = FakeAgentManager()
        scheduler = PriorityScheduler()
        worker = NativeTaskWorker(
            scheduler, manager, rpc=self.rpc,
            agent_permissions={"agent_1": ("get_outdoor_weather",)},
            poll_seconds=3600,
        )
        worker.start()
        try:
            first_deadline = time.monotonic() + 61
            self.engine.tick(now=first_deadline)
            self.assertTrue(worker.poll_once())
            self._wait_one_notification(manager, expected=1, worker=worker)
            first_results = self.rpc.call("notifications", "list")
            self.assertEqual(len(first_results), 1)
            self.assertIn("8.36°C", first_results[0]["text"])

            self.engine.tick(now=first_deadline + 61)
            # Mark previous run finished before the next admission.
            self.assertTrue(worker.poll_once())
            self._wait_one_notification(manager, expected=2, worker=worker)
            second_results = self.rpc.call("notifications", "list")
            self.assertEqual(len(second_results), 2)
            self.assertNotEqual(second_results[0]["run_id"], second_results[1]["run_id"])
            self.assertEqual(len(manager.calls), 2)
            self.assertTrue(all(agent == "agent_1" for agent, _ in manager.calls))

            self.controller.begin_turn()
            try:
                stopped = self.controller.stop(item["task_id"])
            finally:
                self.controller.finish_turn()
            self.assertTrue(stopped["ok"])
            self.engine.tick(now=first_deadline + 122)
            self.assertEqual(len(manager.calls), 2)
        finally:
            worker.close()
            scheduler.close()

    def _wait_one_notification(self, manager, *, expected, worker):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            worker.poll_once()
            if len(manager.calls) >= expected:
                events = self.rpc.call("notifications", "list")
                if len(events) >= expected:
                    return
            time.sleep(0.005)
        self.fail(f"QUERY notifications did not reach count {expected}")

    def test_native_period_argument_types_are_normalized_exactly(self):
        self.controller.enabled = True
        for raw in (60.0, "60", " 60 ", "60.0", 60):
            self.controller.begin_turn()
            try:
                result = self.controller.create(raw, "get_outdoor_weather", "температура")
                self.assertTrue(result["ok"], (raw, result))
                self.assertEqual(result["period_seconds"], 60)
                record = self.rpc.call("tasks", "require", result["task_id"])
                self.assertEqual(record["timer_period_seconds"], 60.0)
            finally:
                self.controller.finish_turn()

        for raw in (True, False, 60.5, float("inf"), float("nan"),
                    "60.5", "abc", "", None, -60, 0, 604801):
            self.controller.begin_turn()
            try:
                before = len(self.rpc.calls)
                result = self.controller.create(raw, "get_outdoor_weather", "температура")
                self.assertFalse(result["ok"], (raw, result))
                self.assertEqual(len(self.rpc.calls), before)
            finally:
                self.controller.finish_turn()

    def test_invalid_inputs_and_rpc_failure_are_not_blindly_retried(self):
        self.controller.enabled = True
        self.controller.begin_turn()
        try:
            for args in ((True, "get_outdoor_weather", "x"),
                         (0, "get_outdoor_weather", "x"),
                         (60, "fake_tool", "x"),
                         (60, "get_outdoor_weather", "")):
                self.assertFalse(self.controller.create(*args)["ok"])
            self.assertEqual(self.rpc.calls, [])
        finally:
            self.controller.finish_turn()

        class BrokenRPC:
            def __init__(self):
                self.calls = 0

            def call(self, *args, **kwargs):
                self.calls += 1
                raise TaskStoreError("Task SYSTEM unavailable at socket: timeout")

        broken = BrokenRPC()
        control = NativeTaskController(broken, {"agent_1": ("get_outdoor_weather",)})
        control.enabled = True
        control.begin_turn()
        try:
            result = control.create(60, "get_outdoor_weather", "получить температуру")
            self.assertFalse(result["ok"])
            self.assertEqual(result["status"], "unknown")
            second = control.create(60, "get_outdoor_weather", "получить температуру")
            self.assertFalse(second["ok"])
            self.assertEqual(broken.calls, 1)
        finally:
            control.finish_turn()


if __name__ == "__main__":
    unittest.main()
