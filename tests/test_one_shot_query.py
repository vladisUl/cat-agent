from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentWorker
from orchestration.assistant_manager import AssistantManagerRuntime
from orchestration.event_store import EventStore
from orchestration.model_client import ChatResponse
from orchestration.pool import AgentPool
from orchestration.prompt_store import PromptStore
from orchestration.skills import SkillBase
from orchestration.system_events import SystemRuntime
from orchestration.tasks import TaskStore


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        return ChatResponse(self.replies.pop(0), None, None, 0.001)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        del messages


class OneShotQueryTest(unittest.TestCase):
    def _runtime(self, root: Path, replies: list[str]) -> AssistantManagerRuntime:
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
            max_file_bytes=4096,
            command_timeout_seconds=2,
        )
        return AssistantManagerRuntime(
            client,  # type: ignore[arg-type]
            SkillBase(prompt_dir / "prompt_base.txt"),
            store,
            AgentPool([worker]),
            SystemRuntime(TaskStore(root / "task.txt")),
            max_steps=8,
            event_store=EventStore(root / "events.json"),
        )

    def test_done_true_is_success_without_publishable_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = self._runtime(Path(temp), ['{"done":true}'])

            result = runtime._execute_task_command(
                [
                    "query_timer.sh",
                    "0",
                    "shell",
                    "Посчитать файлы и сообщить только если количество равно 4",
                ]
            )

            self.assertEqual(
                result,
                "SYSTEM_OK\nЗАПРОС выполнен без результата для сообщения пользователю",
            )

    def test_result_text_is_returned_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = self._runtime(Path(temp), ['{"result":"Количество файлов 4"}'])

            result = runtime._execute_task_command(
                [
                    "query_timer.sh",
                    "0",
                    "shell",
                    "Посчитать файлы и сообщить только если количество равно 4",
                ]
            )

            self.assertEqual(result, "Количество файлов 4")


if __name__ == "__main__":
    unittest.main()
