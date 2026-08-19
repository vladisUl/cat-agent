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


class AssistantManagerTest(unittest.TestCase):
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

    def test_manager_base_is_exact_system_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime, _client = self._runtime(root, ["REPLY\nunused"])
            expected = (root / "prompts" / "sys_prompt_manager.txt").read_text(
                encoding="utf-8"
            ).strip()
            self.assertEqual(runtime.messages[0]["content"].strip(), expected)
            self.assertNotIn("[MANAGER_TOOLS]", runtime.messages[0]["content"])
            self.assertNotIn("[AGENT_EXECUTION_PROTOCOL]", runtime.messages[0]["content"])

    def test_direct_work_uses_same_manager_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = self._runtime(
                Path(temp),
                [
                    "/work#printf helper-ok",
                    "REPLY\nhelper-ok",
                ],
            )

            turn = runtime.user_message("выведи helper-ok")

            self.assertEqual(turn.kind, "reply")
            self.assertEqual(turn.text, "helper-ok")
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[0][-1]["content"], "выведи helper-ok")
            self.assertIn("helper-ok", client.calls[1][-1]["content"])

    def test_sam_is_not_a_special_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = self._runtime(
                Path(temp),
                ["REPLY\nобычный ответ"],
            )

            turn = runtime.user_message("сам посмотри файл")

            self.assertEqual(turn.text, "обычный ответ")
            self.assertEqual(client.calls[0][-1]["content"], "сам посмотри файл")

    def test_external_period_creates_event_bound_query(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, _client = self._runtime(Path(temp), ["REPLY\nunused"])

            result = runtime._execute_task_command(
                [
                    "query_timer.sh",
                    "-1",
                    "shell",
                    "датчик открытия двери",
                    "Проверить состояние двери и вернуть результат.",
                ]
            )

            self.assertIn("SYSTEM_OK", result)
            self.assertIn("task_gpio1", result)
            task = runtime.system_runtime.task_store.require(1)  # type: ignore[union-attr]
            self.assertEqual(task.method, "query")
            self.assertIsNone(task.timer_period_seconds)
            binding = runtime.event_store.resolve("gpio", "task_gpio1")
            self.assertIsNotNone(binding)
            self.assertEqual(binding.task_id, 1)  # type: ignore[union-attr]

            event = runtime.external_event("gpio", "task_gpio1")
            self.assertIsNotNone(event)
            self.assertEqual(event.task_id, 1)  # type: ignore[union-attr]

    def test_invalid_negative_period_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, _client = self._runtime(Path(temp), ["REPLY\nunused"])
            result = runtime._execute_task_command(
                ["task_timer.sh", "-0.5", "shell", "bad", "do bad"]
            )
            self.assertEqual(
                result,
                "SYSTEM_ERROR\nperiod_seconds must be -1, 0 or > 0",
            )


if __name__ == "__main__":
    unittest.main()
