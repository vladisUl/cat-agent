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
        self.calls: list[list[dict[str, str]]] = []
        self.reset_calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self.calls.append([dict(item) for item in messages])
        return ChatResponse(self.replies.pop(0), None, None, 0.001)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        self.reset_calls.append([dict(item) for item in messages])


class OneShotQueryTest(unittest.TestCase):
    def _runtime(
        self,
        root: Path,
        replies: list[str],
    ) -> tuple[AssistantManagerRuntime, FakeClient]:
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
        runtime = AssistantManagerRuntime(
            client,  # type: ignore[arg-type]
            SkillBase(prompt_dir / "prompt_base.txt"),
            store,
            AgentPool([worker]),
            SystemRuntime(TaskStore(root / "task.txt")),
            max_steps=8,
            event_store=EventStore(root / "events.json"),
        )
        return runtime, client

    def test_done_true_has_no_manager_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, _client = self._runtime(Path(temp), ['{"done":true}'])

            result = runtime._execute_task_command(
                [
                    "query_timer.sh",
                    "0",
                    "shell",
                    "Посчитать файлы и сообщить только если количество равно 4",
                ]
            )

            self.assertIsNone(result)

    def test_done_true_ends_human_turn_silently_without_second_manager_tt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            task = (
                "Посчитать количество файлов в рабочем каталоге и сообщить мне "
                "только если их 999999"
            )
            runtime, client = self._runtime(
                Path(temp),
                [
                    f'/work#query_timer.sh 0 shell "{task}"',
                    '{"done":true}',
                ],
            )

            turn = runtime.user_message(f"создай запрос {task.lower()}")

            self.assertEqual(turn.kind, "silent")
            self.assertEqual(turn.text, "")
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.replies, [])
            self.assertEqual(runtime.messages, runtime._base_messages)

    def test_result_text_is_returned_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, _client = self._runtime(
                Path(temp),
                ['{"result":"Количество файлов 4"}'],
            )

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
