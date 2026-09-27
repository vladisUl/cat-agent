from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentState, AgentWorker
from orchestration.manager import ManagerRuntime
from orchestration.model_client import ChatResponse
from orchestration.pool import AgentPool
from orchestration.prompt_store import PromptStore
from orchestration.tool_catalog import build_tool_catalog
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


class ManagerSelfChatTest(unittest.TestCase):
    def _runtime(
        self,
        root: Path,
        replies: list[str],
    ) -> tuple[ManagerRuntime, FakeClient, AgentWorker]:
        prompt_dir = root / "prompts"
        project_root = Path(__file__).resolve().parents[1]
        shutil.copytree(project_root / "prompts", prompt_dir)
        tools_dir = root / "tools"
        shutil.copytree(project_root / "tools", tools_dir)
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
        catalog = build_tool_catalog(tools_dir, ())
        manager = ManagerRuntime(
            client,  # type: ignore[arg-type]
            catalog,
            store,
            AgentPool([worker]),
            SystemRuntime(TaskStore(root / "task.txt")),
            max_steps=8,
        )
        return manager, client, worker

    def test_manager_base_contains_canonical_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager, _client, _worker = self._runtime(root, ["REPLY\nunused"])
            base = manager.messages[0]["content"]

            for name in (
                "cyclic_process",
                "file_divide",
                "mqtt",
                "prognoz",
                "read_pic",
                "shell",
            ):
                raw = (root / "tools" / f"{name}.md").read_text(
                    encoding="utf-8"
                ).strip()
                self.assertIn(raw, base)
                self.assertEqual(base.count(raw), 1)

            self.assertIn("zigbee2mqtt/temp_ulica", base)
            self.assertNotIn("DYNAMIC SKILLS", base)
            self.assertNotIn("[MANAGER_TOOLS]", base)
            self.assertNotIn("[AGENT_EXECUTION_PROTOCOL]", base)

    def test_manager_direct_strips_sam_and_uses_agent_task_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, client, worker = self._runtime(
                Path(temp),
                [
                    "printf self-ok",
                    '{"result":"self-ok"}',
                ],
            )

            turn = manager.user_message("СаМ выведи self-ok")

            self.assertEqual(turn.kind, "reply")
            self.assertEqual(turn.text, "self-ok")
            self.assertEqual(worker.state, AgentState.FREE)
            self.assertEqual(len(client.calls), 2)
            first_tick = json.loads(client.calls[0][-1]["content"])
            self.assertEqual(first_tick, {"task": "выведи self-ok"})
            self.assertNotIn("СаМ", client.calls[0][-1]["content"])
            self.assertNotIn("SELF_MODE", client.calls[0][-1]["content"])
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

    def test_manager_direct_need_uses_same_context_protocol_as_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager, client, worker = self._runtime(
                Path(temp),
                [
                    '{"need":"Нужно имя файла."}',
                    "cat answer.txt",
                    '{"result":"42"}',
                ],
            )
            workspace = worker.workspace
            (workspace / "answer.txt").write_text("42\n", encoding="utf-8")

            first = manager.user_message("сам прочитай нужный файл")
            self.assertEqual(first.kind, "ask")
            self.assertEqual(first.text, "Нужно имя файла.")
            self.assertEqual(worker.state, AgentState.FREE)

            second = manager.user_message("answer.txt")
            self.assertEqual(second.kind, "reply")
            self.assertEqual(second.text, "42")
            context_tick = json.loads(client.calls[1][-1]["content"])
            self.assertEqual(context_tick, {"context": "answer.txt"})
            self.assertEqual(len(client.reset_calls), 1)

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
