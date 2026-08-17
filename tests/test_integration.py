from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentWorker
from orchestration.manager import ManagerRuntime
from orchestration.model_client import ChatResponse
from orchestration.pool import AgentPool
from orchestration.prompt_store import PromptStore
from orchestration.skills import SkillBase


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.reset_calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self.calls.append([dict(item) for item in messages])
        content = self.replies.pop(0)
        return ChatResponse(content, None, None, 0.001)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        self.reset_calls.append([dict(item) for item in messages])


class IntegrationTest(unittest.TestCase):
    def test_manager_agent_manager_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "alpha.txt").write_text("alpha\n", encoding="utf-8")

            client = FakeClient(
                [
                    "DELEGATE shell\nПосмотри список файлов и верни имена.",
                    "ls",
                    '{"result":"В каталоге есть alpha.txt."}',
                    "REPLY\nВ рабочем каталоге есть alpha.txt.",
                ]
            )
            store = PromptStore(prompt_dir, 3)
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
            manager = ManagerRuntime(
                client,  # type: ignore[arg-type]
                SkillBase(prompt_dir / "prompt_base.txt"),
                store,
                AgentPool([worker]),
                max_steps=6,
            )
            turn = manager.user_message("Какие файлы лежат в рабочем каталоге?")
            self.assertEqual(turn.kind, "reply")
            self.assertIn("alpha.txt", turn.text)

            built = (prompt_dir / "prompt_agent_1.txt").read_text(encoding="utf-8")
            self.assertIn('"name": "shell"', built)
            self.assertIn("Посмотри список файлов", built)

            first_agent_call = client.calls[1]
            self.assertEqual(
                [message["role"] for message in first_agent_call],
                ["system", "user"],
            )
            self.assertNotIn("READY", str(first_agent_call))
            self.assertNotIn("Посмотри список файлов", first_agent_call[0]["content"])
            first_tick = json.loads(first_agent_call[1]["content"])
            self.assertEqual(first_tick, {"task": "Посмотри список файлов и верни имена."})

            second_agent_call = client.calls[2]
            self.assertEqual(second_agent_call[-1]["role"], "user")
            self.assertIn("[exit_code=0]", second_agent_call[-1]["content"])
            self.assertIn("alpha.txt", second_agent_call[-1]["content"])
            self.assertEqual(len(client.reset_calls), 2)

    def test_repeated_matching_agent_tasks_reuse_clean_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()

            client = FakeClient(
                [
                    '{"result":"Первая задача выполнена."}',
                    '{"result":"Вторая задача выполнена."}',
                ]
            )
            store = PromptStore(prompt_dir, 3)
            store.validate()
            worker = AgentWorker(
                "agent1",
                client,  # type: ignore[arg-type]
                store,
                workspace,
                max_steps=4,
                max_file_bytes=1024,
                command_timeout_seconds=2,
            )
            mqtt = SkillBase(prompt_dir / "prompt_base.txt").require(("mqtt",))

            first = worker.start("Посмотри температуру на улице.", mqtt)
            self.assertEqual(first.status, "OK")
            self.assertEqual(worker.state.value, "FREE")

            first_call = client.calls[0]
            second = worker.start("Посмотри температуру в аквариуме.", mqtt)
            self.assertEqual(second.status, "OK")
            second_call = client.calls[1]

            self.assertEqual(first_call[:1], second_call[:1])
            self.assertEqual(len(second_call), 2)
            self.assertEqual(second_call[-1]["role"], "user")
            self.assertIn("температуру в аквариуме", second_call[-1]["content"])
            self.assertNotIn("Первая задача выполнена", str(second_call))
            self.assertEqual(len(client.reset_calls), 2)
            self.assertEqual(client.reset_calls[0], first_call[:1])

    def test_manager_replies_start_next_request_from_clean_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()

            client = FakeClient(
                [
                    "REPLY\nПервый ответ.",
                    "REPLY\nВторой ответ.",
                ]
            )
            store = PromptStore(prompt_dir, 1)
            store.validate()
            worker = AgentWorker(
                "agent1",
                client,  # type: ignore[arg-type]
                store,
                workspace,
                max_steps=4,
                max_file_bytes=1024,
                command_timeout_seconds=2,
            )
            manager = ManagerRuntime(
                client,  # type: ignore[arg-type]
                SkillBase(prompt_dir / "prompt_base.txt"),
                store,
                AgentPool([worker]),
                max_steps=4,
            )

            first = manager.user_message("Первая задача")
            second = manager.user_message("Вторая задача")

            self.assertEqual(first.kind, "reply")
            self.assertEqual(second.kind, "reply")
            self.assertEqual(client.calls[0][:1], client.calls[1][:1])
            self.assertEqual(len(client.calls[1]), 2)
            self.assertEqual(client.calls[1][0]["role"], "system")
            self.assertEqual(client.calls[1][1]["role"], "user")
            self.assertIn("Вторая задача", client.calls[1][-1]["content"])
            self.assertNotIn("Первая задача", str(client.calls[1]))
            self.assertEqual(len(client.reset_calls), 2)

    def test_need_ask_continue_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "answer.txt").write_text("42\n", encoding="utf-8")

            client = FakeClient(
                [
                    "DELEGATE shell\nПрочитай файл, имя которого должен сообщить пользователь.",
                    '{"need":"Нужно имя файла."}',
                    "ASK\nКак называется файл?",
                    "CONTINUE agent1\nИмя файла: answer.txt",
                    "cat answer.txt",
                    '{"result":"В файле записано 42."}',
                    "REPLY\nВ файле записано 42.",
                ]
            )
            store = PromptStore(prompt_dir, 3)
            store.validate()
            worker = AgentWorker(
                "agent1",
                client,  # type: ignore[arg-type]
                store,
                workspace,
                max_steps=8,
                max_file_bytes=1024,
                command_timeout_seconds=2,
            )
            manager = ManagerRuntime(
                client,  # type: ignore[arg-type]
                SkillBase(prompt_dir / "prompt_base.txt"),
                store,
                AgentPool([worker]),
                max_steps=8,
            )

            first = manager.user_message("Прочитай нужный файл.")
            self.assertEqual(first.kind, "ask")
            self.assertEqual(worker.state.value, "WAITING")
            self.assertEqual(len(client.reset_calls), 0)

            second = manager.user_message("answer.txt")
            self.assertEqual(second.kind, "reply")
            self.assertIn("42", second.text)
            self.assertEqual(worker.state.value, "FREE")
            self.assertEqual(len(client.reset_calls), 2)

            continue_agent_call = client.calls[4]
            context_tick = json.loads(continue_agent_call[-1]["content"])
            self.assertEqual(context_tick, {"context": "Имя файла: answer.txt"})


if __name__ == "__main__":
    unittest.main()
