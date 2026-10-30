"""Unified native delegation: one-shot and periodic use identical AGENT text."""
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
from orchestration_tools.delegation import DelegationBroker, create_delegate_tool
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


class FakeManager:
    def __init__(self):
        self.calls = []

    def agent_status(self):
        return [
            {"agent_id": "agent_1", "state": "FREE"},
            {"agent_id": "agent_2", "state": "FREE"},
        ]

    def run_agent(self, agent_id, text):
        self.calls.append((agent_id, text))
        return AgentOutcome(agent_id, AgentState.FREE, "Сделано", 250)




class UnifiedDelegationTests(unittest.TestCase):
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
            self.rpc, {"agent_1": ("get_outdoor_weather", "shell"),
                       "agent_2": ("get_outdoor_weather",)},
        )
        self.broker = DelegationBroker(("agent_1", "agent_2"), self.controller)
        self.tool = create_delegate_tool(self.broker)
        self.management = {
            t.__name__: t for t in create_task_tools(self.controller)
        }
        self.controller.enabled = True
        self.broker.enabled = True

    def _begin(self):
        self.controller.begin_turn()
        self.broker.begin_turn()

    def _end(self):
        self.broker.finish_turn()
        self.controller.finish_turn()

    def test_zero_is_original_one_shot_delegation(self):
        for interval in (0, "0", "0.0", 0.0):
            self._begin()
            try:
                result = self.tool("agent_1", " MQTT -> shell -> файл ", interval)
                self.assertTrue(result["ok"], result)
                self.assertEqual(result["status"], "queued")
                pending, = self.broker.take_pending()
                self.assertEqual(pending.agent_id, "agent_1")
                self.assertEqual(pending.task, "MQTT -> shell -> файл")
                self.assertFalse(self.rpc.calls)
            finally:
                self._end()

    def test_periodic_persists_unmodified_multistep_instruction_and_target(self):
        original = ("Получить температуру через MQTT, если файла нет создать "
                    "temp.txt, дописать строку через shell.")
        self._begin()
        try:
            result = self.tool("agent_1", original, "60")
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["status"], "scheduled")
            self.assertEqual(result["period_seconds"], 60)
            self.assertEqual(result["agent_id"], "agent_1")
            self.assertEqual(self.broker.take_pending(), ())
            task = self.rpc.call("tasks", "require", result["task_id"])
            self.assertEqual(task["text"], original)
            self.assertEqual(task["skills"], [])
            self.assertEqual(task["agent_id"], "agent_1")
            self.assertEqual(task["method"], "query")
            self.assertEqual(task["executor"], "litert")
            self.assertEqual(task["timer_period_seconds"], 60.0)
            self.assertEqual(self.tool("agent_1", original, "60")["ok"], False)
            self.assertEqual(
                len([c for c in self.rpc.calls if c[1] == "create_periodic_task"]), 1,
            )
            listed = self.management["list_periodic_queries"]()
            self.assertEqual(listed["tasks"][0]["agent_id"], "agent_1")
            stopped = self.management["stop_periodic_query"](str(result["task_id"]))
            self.assertTrue(stopped["ok"])
            self.assertFalse(stopped["enabled"])
        finally:
            self._end()

    def test_delete_removes_stopped_task_and_frees_identifier(self):
        self._begin()
        try:
            result = self.tool("agent_1", "Проверять температуру", 60)
            task_id = result["task_id"]
            stopped = self.management["stop_periodic_query"](task_id)
            self.assertEqual(stopped, {"ok": True, "task_id": task_id, "enabled": False})
            self.assertIsNotNone(self.rpc.call("tasks", "get", task_id))
            deleted = self.management["delete_periodic_query"](str(task_id))
            self.assertEqual(deleted, {"ok": True, "task_id": task_id, "deleted": True})
            self.assertIsNone(self.rpc.call("tasks", "get", task_id))
            self.assertFalse(any(t["task_id"] == task_id
                                 for t in self.management["list_periodic_queries"]()["tasks"]))
            self.assertFalse(any(t.task_id == task_id
                                 for t in self.engine.runtime.task_timer_snapshot()))
            self.assertFalse(self.management["delete_periodic_query"](task_id)["ok"])
        finally:
            self._end()
        # Deletion frees the numeric slot; stop alone never does.
        self._begin()
        try:
            recreated = self.tool("agent_1", "Новая задача", 90)
            self.assertTrue(recreated["ok"], recreated)
            self.assertEqual(recreated["task_id"], task_id)
        finally:
            self._end()

    def test_delete_denies_non_native_tasks_and_invalid_or_inactive_requests(self):
        tool = self.management["delete_periodic_query"]
        self.assertEqual(tool(1)["error"], "task_management_not_in_active_turn")
        self._begin()
        try:
            for identifier in (0, -1, True, 1.5, "not-an-id"):
                self.assertEqual(tool(identifier)["error"], "task_id must be a positive integer")
            foreign = self.engine.runtime.create_periodic_task(
                "Other CORE task", "do stuff", ("shell",), 60,
                method="query", executor="openai",
            )
            self.assertEqual(tool(foreign.task_id)["error"],
                             "native periodic QUERY not found")
            self.assertIsNotNone(self.rpc.call("tasks", "get", foreign.task_id))
        finally:
            self._end()

    def test_delete_active_task_revokes_future_runs(self):
        self._begin()
        try:
            result = self.tool("agent_1", "Проверять периодически", 60)
            task_id = result["task_id"]
            task = self.rpc.call("tasks", "require", task_id)
            timer = self.engine.runtime._task_timers[task_id]
            self.engine.tick(timer.next_fire_monotonic + 0.01)
            run_id = next(rid for rid, run in self.engine.runs.items()
                          if run["event"]["task_id"] == task_id)
            removed = self.management["delete_periodic_query"](task_id)
            self.assertTrue(removed["ok"])
            self.engine.tick(timer.next_fire_monotonic + 61)
            self.assertIsNone(self.rpc.call("tasks", "get", task_id))
            self.assertEqual(self.engine.runs[run_id]["state"], "cancelled")
            self.assertEqual(len(self.engine.runs), 1)
        finally:
            self._end()

    def test_periodic_executes_same_text_twice_on_selected_agent_and_stops(self):
        text = "Прочитать MQTT, добавить строку в temp.txt через shell"
        self._begin()
        try:
            result = self.tool("agent_1", text, 60)
        finally:
            self._end()
        self.assertTrue(result["ok"], result)
        manager = FakeManager()
        scheduler = PriorityScheduler()
        worker = NativeTaskWorker(
            scheduler, manager, rpc=self.rpc,
            agent_permissions={
                "agent_1": ("get_outdoor_weather", "shell"),
                "agent_2": ("get_outdoor_weather",),
            }, poll_seconds=3600,
        )
        worker.start()
        try:
            first = time.monotonic() + 61
            self.engine.tick(now=first)
            self.assertTrue(worker.poll_once())
            self._await_reports(worker, manager, 1)
            self.engine.tick(now=first + 61)
            self.assertTrue(worker.poll_once())
            self._await_reports(worker, manager, 2)
            self.assertEqual(manager.calls, [
                ("agent_1", text), ("agent_1", text),
            ])
            reports = self.rpc.call("notifications", "list")
            self.assertEqual(len(reports), 2)
            self.assertNotEqual(reports[0]["run_id"], reports[1]["run_id"])
            self._begin()
            try:
                self.assertTrue(self.management["stop_periodic_query"](
                    result["task_id"])["ok"])
            finally:
                self._end()
            self.engine.tick(now=first + 122)
            self.assertEqual(len(manager.calls), 2)
        finally:
            worker.close()
            scheduler.close()

    def _await_reports(self, worker, manager, number):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            worker.poll_once()
            if len(manager.calls) >= number:
                if len(self.rpc.call("notifications", "list")) >= number:
                    return
            time.sleep(0.005)
        self.fail(f"no {number} durable notifications")

    def test_unknown_agent_invalid_period_missing_system_and_warm_guard(self):
        self.broker.enabled = False
        self.assertFalse(self.tool("agent_1", "test", 60)["ok"])
        self.broker.enabled = True
        self.assertEqual(self.tool("agent_1", "test", 60)["error"],
                         "no_active_manager_turn")
        self._begin()
        try:
            for period in (True, -1, 60.5, "60.5", "abc", 604801):
                self.assertFalse(self.tool("agent_1", "test", period)["ok"])
            self.assertEqual(self.tool("agent_3", "task", 60)["error"],
                             "unknown_agent")
            self.assertFalse(self.tool("agent_1", "", 60)["ok"])
            self.assertFalse(self.rpc.calls)
        finally:
            self._end()
        self.broker.periodic_controller = None
        self._begin()
        try:
            result = self.tool("agent_1", "test", 60)
            self.assertEqual(result["error"], "periodic_TASK_SYSTEM_not_configured")
        finally:
            self._end()

    def test_no_duplicate_after_unknown_rpc_reply(self):
        class LostRPC:
            def __init__(self):
                self.calls = 0

            def call(self, *args, **kwargs):
                self.calls += 1
                raise TaskStoreError("Task SYSTEM unavailable at socket: timeout")

        rpc = LostRPC()
        controller = NativeTaskController(rpc, {"agent_1": ("shell",)})
        broker = DelegationBroker(("agent_1",), controller)
        controller.enabled = broker.enabled = True
        controller.begin_turn()
        broker.begin_turn()
        try:
            first = broker.enqueue("agent_1", "read then write", "60")
            self.assertEqual(first["status"], "unknown")
            again = broker.enqueue("agent_1", "read then write", "60")
            self.assertFalse(again["ok"])
            self.assertEqual(rpc.calls, 1)
        finally:
            broker.finish_turn()
            controller.finish_turn()


if __name__ == "__main__":
    unittest.main()
