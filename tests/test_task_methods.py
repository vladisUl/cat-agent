from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from cat_agent.agent import AgentWorker
from cat_agent.manager import ManagerRuntime
from cat_agent.model_client import ChatResponse
from cat_agent.pool import AgentPool
from cat_agent.prompt_store import PromptStore
from cat_agent.skills import SkillBase
from cat_agent.system_events import SystemEvent, SystemRuntime
from cat_agent.tasks import TaskStore


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.reset_calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self.calls.append([dict(item) for item in messages])
        return ChatResponse(self.replies.pop(0), None, None, 0.001)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        self.reset_calls.append([dict(item) for item in messages])


class TaskMethodTest(unittest.TestCase):
    def _runtime(self, root: Path, replies: list[str]) -> tuple[ManagerRuntime, SystemRuntime, FakeClient]:
        prompt_dir = root / "prompts"
        shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
        workspace = root / "workspace"
        workspace.mkdir()
        client = FakeClient(replies)
        store = PromptStore(prompt_dir, 1)
        store.validate()
        worker = AgentWorker(
            "agent1",
            client,  # type: ignore[arg-type]
            store,
            workspace,
            max_steps=6,
            max_file_bytes=1024,
            command_timeout_seconds=2,
        )
        system = SystemRuntime(TaskStore(root / "task.txt"))
        manager = ManagerRuntime(
            client,  # type: ignore[arg-type]
            SkillBase(prompt_dir / "prompt_base.txt"),
            store,
            AgentPool([worker]),
            system,
            max_steps=6,
        )
        return manager, system, client

    def test_autonomous_task_returns_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, system, client = self._runtime(root, ["DONE"])
            task = system.create_periodic_task(
                "действие",
                "Выполнить действие.",
                ("shell",),
                60.0,
                method="task",
            )
            turn = manager.system_event(
                SystemEvent("timer", f"task:{task.task_id}", "", 1.0, task.task_id)
            )
            self.assertEqual(turn.kind, "silent")
            self.assertEqual(turn.text, "")
            self.assertEqual(len(client.calls), 1)
            self.assertIn("[METHOD]\nTASK\n[/METHOD]", client.calls[0][-1]["content"])

    def test_autonomous_query_returns_value_without_manager_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, system, client = self._runtime(root, ["DONE\nОК"])
            task = system.create_periodic_task(
                "проверка",
                'Проверить состояние. Вернуть "ОК" или "Авария".',
                ("shell",),
                60.0,
                method="query",
            )
            turn = manager.system_event(
                SystemEvent("timer", f"task:{task.task_id}", "", 1.0, task.task_id)
            )
            self.assertEqual(turn.kind, "reply")
            self.assertEqual(turn.text, "ОК")
            self.assertEqual(len(client.calls), 1)
            self.assertIn("[METHOD]\nQUERY\n[/METHOD]", client.calls[0][-1]["content"])

    def test_query_rejects_empty_done_until_value_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, system, client = self._runtime(root, ["DONE", "DONE\nАвария"])
            task = system.create_periodic_task(
                "проверка",
                "Проверить состояние и вернуть результат.",
                ("shell",),
                60.0,
                method="query",
            )
            turn = manager.system_event(
                SystemEvent("timer", f"task:{task.task_id}", "", 1.0, task.task_id)
            )
            self.assertEqual(turn.kind, "reply")
            self.assertEqual(turn.text, "Авария")
            self.assertEqual(len(client.calls), 2)
            self.assertIn("QUERY requires a non-empty return value", client.calls[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
