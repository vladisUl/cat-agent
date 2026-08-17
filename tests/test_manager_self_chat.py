from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from cat_agent.agent import AgentState, AgentWorker
from cat_agent.manager import ManagerRuntime
from cat_agent.model_client import ChatResponse
from cat_agent.pool import AgentPool
from cat_agent.prompt_store import PromptStore
from cat_agent.protocol import ManagerAction, parse_manager_output
from cat_agent.skills import SkillBase
from cat_agent.system_events import SystemRuntime
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


class ManagerSelfChatTest(unittest.TestCase):
    def _runtime(
        self,
        root: Path,
        replies: list[str],
    ) -> tuple[ManagerRuntime, FakeClient, AgentWorker]:
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
        manager = ManagerRuntime(
            client,  # type: ignore[arg-type]
            SkillBase(prompt_dir / "prompt_base.txt"),
            store,
            AgentPool([worker]),
            SystemRuntime(TaskStore(root / "task.txt")),
            max_steps=8,
        )
        return manager, client, worker

    def test_direct_command_protocol_is_only_enabled_by_runtime(self) -> None:
        command = parse_manager_output("printf ok", allow_command=True)
        self.assertEqual(command.action, ManagerAction.COMMAND)
        self.assertEqual(command.command, "printf ok")
        self.assertIsNotNone(parse_manager_output("printf ok").error)

    def test_manager_base_contains_all_tool_instructions_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, _client, _worker = self._runtime(Path(temp), ["REPLY\nunused"])
            base = manager.messages[0]["content"]
            self.assertIn("[MANAGER_TOOLS]", base)
            self.assertIn('"name": "shell"', base)
            self.assertIn('"name": "mqtt"', base)
            self.assertIn("broker_host: 192.168.0.21", base)

    def test_manager_self_strips_sam_and_executes_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, client, worker = self._runtime(
                Path(temp),
                [
                    "printf self-ok",
                    "REPLY\nself-ok",
                ],
            )

            turn = manager.user_message("СаМ выведи self-ok")

            self.assertEqual(turn.kind, "reply")
            self.assertEqual(turn.text, "self-ok")
            self.assertEqual(worker.state, AgentState.FREE)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(
                client.calls[0][-1]["content"],
                "[SELF_MODE]\nвыведи self-ok",
            )
            self.assertNotIn("СаМ", client.calls[0][-1]["content"])
            self.assertNotIn("SELF_READY", client.calls[1][-1]["content"])
            self.assertIn("self-ok", client.calls[1][-1]["content"])
            self.assertEqual(len(client.reset_calls), 1)

    def test_sam_must_be_the_exact_first_word(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, client, _worker = self._runtime(
                Path(temp),
                ["REPLY\nобычный режим"],
            )

            turn = manager.user_message("самолет летит")

            self.assertEqual(turn.text, "обычный режим")
            self.assertEqual(client.calls[0][-1]["content"], "самолет летит")
            self.assertNotIn("[SELF_MODE]", client.calls[0][-1]["content"])

    def test_self_rejects_delegate_and_retries_with_direct_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, client, worker = self._runtime(
                Path(temp),
                [
                    "DELEGATE shell\nвыведи self-ok",
                    "printf self-ok",
                    "REPLY\nself-ok",
                ],
            )

            turn = manager.user_message("сам выведи self-ok")

            self.assertEqual(turn.text, "self-ok")
            self.assertEqual(worker.state, AgentState.FREE)
            self.assertIn("SELF_MODE forbids DELEGATE", client.calls[1][-1]["content"])

    def test_chat_preserves_context_until_end_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, client, _worker = self._runtime(
                Path(temp),
                [
                    "REPLY\nЧат создан",
                    "REPLY\nПервая",
                    "REPLY\nВторая помнит Первую",
                    "REPLY\nЧат закрыт",
                ],
            )

            opened = manager.user_message("Чат")
            self.assertEqual(opened.text, "Чат создан")
            self.assertEqual(len(client.reset_calls), 0)

            first = manager.user_message("скажи Первая")
            self.assertEqual(first.text, "Первая")
            self.assertEqual(len(client.reset_calls), 0)

            second = manager.user_message("что было до этого?")
            self.assertEqual(second.text, "Вторая помнит Первую")
            self.assertTrue(
                any(
                    message["role"] == "assistant" and message["content"] == "REPLY\nПервая"
                    for message in client.calls[2]
                )
            )
            self.assertEqual(len(client.reset_calls), 0)

            closed = manager.user_message("Конец чата")
            self.assertEqual(closed.text, "Чат закрыт")
            self.assertEqual(len(client.reset_calls), 1)
            self.assertEqual(manager.messages, manager._base_messages)


if __name__ == "__main__":
    unittest.main()
