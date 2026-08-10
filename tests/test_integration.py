from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from cat_agent.agent import AgentWorker
from cat_agent.manager import ManagerRuntime
from cat_agent.model_client import ChatResponse
from cat_agent.pool import AgentPool
from cat_agent.prompt_store import AGENT_BOOTSTRAP_ACK, PromptStore
from cat_agent.skills import SkillBase


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self.calls.append([dict(item) for item in messages])
        content = self.replies.pop(0)
        return ChatResponse(content, None, None, 0.001)


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
                    "DONE\nВ каталоге есть alpha.txt.",
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
            self.assertIn("[SKILL shell]", built)
            self.assertIn("Посмотри список файлов", built)

            first_agent_call = client.calls[1]
            self.assertEqual(
                [message["role"] for message in first_agent_call[:4]],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(first_agent_call[2]["content"], AGENT_BOOTSTRAP_ACK)
            self.assertNotIn("[TASK]", first_agent_call[1]["content"])
            self.assertIn("[TASK]", first_agent_call[3]["content"])
            self.assertIn("Посмотри список файлов", first_agent_call[3]["content"])

    def test_repeated_matching_agent_tasks_append_to_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()

            client = FakeClient(
                [
                    "DONE\nПервая задача выполнена.",
                    "DONE\nВторая задача выполнена.",
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

            self.assertEqual(second_call[: len(first_call)], first_call)
            self.assertEqual(second_call[len(first_call)]["role"], "user")
            self.assertIn("температуру в аквариуме", second_call[len(first_call)]["content"])

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
                    "NEED\nНужно имя файла.",
                    "ASK\nКак называется файл?",
                    "CONTINUE agent1\nИмя файла: answer.txt",
                    "cat answer.txt",
                    "DONE\nВ файле записано 42.",
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

            second = manager.user_message("answer.txt")
            self.assertEqual(second.kind, "reply")
            self.assertIn("42", second.text)
            self.assertEqual(worker.state.value, "FREE")


if __name__ == "__main__":
    unittest.main()
