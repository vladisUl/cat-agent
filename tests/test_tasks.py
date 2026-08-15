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
            second = store.create("проверка системы", "Проверить состояние системы.")

            self.assertEqual(first.task_id, 1)
            self.assertEqual(second.task_id, 2)
            self.assertTrue(path.is_file())

            restored = TaskStore(path)
            self.assertEqual(restored.list(), (first, second))
            self.assertEqual(
                restored.status_text(),
                "TASK 1 ловля котов\nTASK 2 проверка системы",
            )

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
            system.set_task_handler(received.append)

            activation = system.activate_task(
                task.task_id,
                source="timer",
                name="periodic",
                now=123.0,
            )

            self.assertEqual(received, [activation])
            self.assertEqual(activation.task, task)
            self.assertEqual(activation.source, "timer")
            self.assertEqual(activation.name, "periodic")
            self.assertEqual(activation.created_monotonic, 123.0)


if __name__ == "__main__":
    unittest.main()
