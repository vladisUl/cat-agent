from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from orchestration.model_client import ChatResponse
from orchestration.read_partly import read_partly
from orchestration.workspace_command_runtime import CommandRuntime


class FakeClient:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.reset_calls: list[list[dict[str, str]]] = []
        self.children: list[FakeClient] = []
        self.fork_inherit_base: list[bool] = []
        self.closed = False

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self.calls.append([dict(item) for item in messages])
        return ChatResponse(self.replies.pop(0), None, None, 0.001)

    def fork(
        self,
        label: str,
        *,
        inherit_base: bool = True,
    ) -> "FakeClient":
        del label
        self.fork_inherit_base.append(inherit_base)
        child = FakeClient(self.replies)
        self.children.append(child)
        return child

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        self.reset_calls.append([dict(item) for item in messages])

    def close(self) -> None:
        self.closed = True


class ReadPartlyTest(unittest.TestCase):
    def test_returns_combined_text_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "story.txt").write_text(
                "AAA first line\nAAA second line\n"
                "BBB first line\nBBB second line\n",
                encoding="utf-8",
            )
            runtime = CommandRuntime(
                root,
                ("read_partly",),
                max_file_bytes=4096,
                timeout_seconds=2,
            )
            client = FakeClient(["first-summary", "second-summary"])

            result = read_partly(
                "read_partly.sh story.txt -n 2",
                runtime,
                client,
                task_text="Прочитай весь рассказ и кратко перескажи.",
            )

            self.assertEqual(result, "first-summary\nsecond-summary")
            self.assertEqual(client.fork_inherit_base, [False])
            self.assertEqual(len(client.children), 1)

            child = client.children[0]
            self.assertEqual(len(child.calls), 2)
            first_prompt = child.calls[0][-1]["content"]
            second_prompt = child.calls[1][-1]["content"]
            self.assertIn("AAA first line", first_prompt)
            self.assertNotIn("BBB first line", first_prompt)
            self.assertIn("BBB first line", second_prompt)
            self.assertNotIn("AAA first line", second_prompt)
            self.assertEqual(len(child.reset_calls), 2)
            self.assertTrue(child.closed)


    def test_decodes_windows_1251_without_replacement_characters(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            source = "Привет, это русский текст.\nВторая строка.\n"
            (data / "legacy.txt").write_bytes(source.encode("cp1251"))
            runtime = CommandRuntime(
                root,
                ("read_partly",),
                max_file_bytes=4096,
                timeout_seconds=2,
            )
            client = FakeClient(["готово"])

            result = read_partly(
                "read_partly.sh legacy.txt -n 1",
                runtime,
                client,
                task_text="Прочитай текст.",
            )

            self.assertEqual(result, "готово")
            prompt = client.children[0].calls[0][-1]["content"]
            self.assertIn("Привет, это русский текст.", prompt)
            self.assertNotIn("�", prompt)

    def test_requires_assignment_and_valid_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "story.txt").write_text("text\n", encoding="utf-8")
            runtime = SimpleNamespace(root=root, skill_names={"shell"})
            client = FakeClient(["unused"])

            self.assertIn(
                "not assigned",
                read_partly(
                    "read_partly.sh story.txt -n 1",
                    runtime,
                    client,
                    task_text="Прочитай.",
                ),
            )
            runtime.skill_names = {"read_partly"}
            self.assertIn(
                "positive integer",
                read_partly(
                    "read_partly.sh story.txt -n 0",
                    runtime,
                    client,
                    task_text="Прочитай.",
                ),
            )


if __name__ == "__main__":
    unittest.main()
