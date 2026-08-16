from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cat_agent.system_events import SystemRuntime, TaskActivation
from cat_agent.tasks import TaskStore, TaskStoreError


class TaskStoreTest(unittest.TestCase):
    def test_tasks_survive_store_reload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task.txt"
            store = TaskStore(path)

            first = store.create("ловля котов", "Искать котов и копировать результат.")
            second = store.create(
                "проверка системы",
                "Проверить состояние системы.",
                method="query",
            )

            self.assertEqual(first.task_id, 1)
            self.assertEqual(first.method, "task")
            self.assertEqual(second.task_id, 2)
            self.assertEqual(second.method, "query")
            self.assertTrue(path.is_file())

            restored = TaskStore(path)
            self.assertEqual(restored.list(), (first, second))
            self.assertEqual(
                restored.status_text(),
                "TASK 1 ловля котов\nTASK 2 проверка системы",
            )

    def test_old_record_without_method_defaults_to_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task.txt"
            path.write_text(
                '{"task":1,"description":"old","text":"work","skills":["shell"],'
                '"timer_period_seconds":60.0,"enabled":false}\n',
                encoding="utf-8",
            )
            restored = TaskStore(path).require(1)
            self.assertEqual(restored.method, "task")
            self.assertFalse(restored.enabled)

    def test_rejects_unknown_method(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir) / "task.txt")
            with self.assertRaises(TaskStoreError):
                store.create("bad", "work", method="report")

    def test_task_ids_are_system_assigned_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir) / "task.txt", max_tasks=2)
            store.create("one", "first")
            store.create("two", "second")

            with self.assertRaises(TaskStoreError):
                store.create("three", "third")

            self.assertTrue(store.delete(1))
            replacement = store.create("replacement", "new first")
            self.assertEqual(replacement.task_id, 1)

    def test_system_resolves_task_before_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir) / "task.txt")
            task = store.create("ловля котов", "Искать котов.")
            system = SystemRuntime(store)
            received: list[TaskActivation] = []

            def handler(activation: TaskActivation) -> None:
                received.append(activation)
                return None

            system.set_task_handler(handler)

            result = system.activate_task(
                task.task_id,
                source="timer",
                name="periodic",
                now=123.0,
            )

            self.assertIsNone(result)
            self.assertEqual(len(received), 1)
            activation = received[0]
            self.assertEqual(activation.task, task)
            self.assertEqual(activation.source, "timer")
            self.assertEqual(activation.name, "periodic")
            self.assertEqual(activation.created_monotonic, 123.0)

    def test_system_returns_query_handler_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = TaskStore(Path(temp_dir) / "task.txt")
            task = store.create("проверка", "Проверить файл.", method="query")
            system = SystemRuntime(store)
            system.set_task_handler(lambda activation: "ОК")

            result = system.activate_task(task.task_id, source="timer")
            self.assertEqual(result, "ОК")

    def test_periodic_query_survives_restart_but_countdown_waits_for_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task.txt"
            first_system = SystemRuntime(TaskStore(path))
            task = first_system.create_periodic_task(
                "проверка котов",
                "Проверить котов и вернуть состояние.",
                ("shell",),
                60.0,
                method="query",
            )

            stored = TaskStore(path).require(task.task_id)
            self.assertEqual(stored.method, "query")
            self.assertEqual(stored.skills, ("shell",))
            self.assertEqual(stored.timer_period_seconds, 60.0)
            self.assertTrue(stored.enabled)

            restarted = SystemRuntime(TaskStore(path))
            timers = restarted.task_timer_snapshot()
            self.assertEqual(len(timers), 1)
            self.assertEqual(timers[0].task_id, task.task_id)
            self.assertTrue(timers[0].enabled)
            self.assertIsNone(timers[0].next_fire_monotonic)
            self.assertEqual(restarted.poll_due(1000000.0), ())

            restarted.arm_task_timers(now=100.0)
            timer = restarted.task_timer_snapshot()[0]
            self.assertEqual(timer.next_fire_monotonic, 160.0)

            events = restarted.poll_due(160.0)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].task_id, task.task_id)
            self.assertEqual(events[0].source, "timer")
            self.assertEqual(events[0].task, "")

    def test_stopped_periodic_task_stays_stopped_after_restart_and_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "task.txt"
            system = SystemRuntime(TaskStore(path))
            task = system.create_periodic_task(
                "ловля котов",
                "Искать котов.",
                ("shell",),
                60.0,
            )
            system.stop_task(task.task_id)

            restarted = SystemRuntime(TaskStore(path))
            timer = restarted.task_timer_snapshot()[0]
            self.assertFalse(timer.enabled)
            self.assertIsNone(timer.next_fire_monotonic)

            restarted.arm_task_timers(now=100.0)
            timer = restarted.task_timer_snapshot()[0]
            self.assertFalse(timer.enabled)
            self.assertIsNone(timer.next_fire_monotonic)


if __name__ == "__main__":
    unittest.main()
