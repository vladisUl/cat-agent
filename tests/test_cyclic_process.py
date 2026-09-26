from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentWorker
from orchestration.model_client import ChatResponse
from orchestration.prompt_store import PromptStore
from orchestration.skill_script import run_skill_script
from orchestration.skills import Skill
from orchestration.workspace_command_runtime import CommandRuntime


class FakeClient:
    def __init__(self, replies: list[str], child_replies: list[str] | None = None) -> None:
        self.replies = list(replies)
        self.child_replies = list(child_replies or [])
        self.calls: list[list[dict[str, str]]] = []
        self.reset_calls: list[list[dict[str, str]]] = []
        self.children: list[FakeClient] = []
        self.fork_inherit_base: list[bool] = []
        self.closed = False

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        self.calls.append([dict(item) for item in messages])
        return ChatResponse(self.replies.pop(0), None, None, 0.001)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        self.reset_calls.append([dict(item) for item in messages])

    def fork(
        self,
        label: str,
        *,
        inherit_base: bool = True,
    ) -> "FakeClient":
        del label
        self.fork_inherit_base.append(inherit_base)
        child = FakeClient(self.child_replies)
        self.children.append(child)
        return child

    def close(self) -> None:
        self.closed = True


class CyclicProcessTest(unittest.TestCase):
    @staticmethod
    def _workspace(root: Path) -> Path:
        workspace = root / "workspace"
        data = workspace / "data"
        data.mkdir(parents=True)
        (data / "syslog_1").write_text("AAA first fragment\n", encoding="utf-8")
        (data / "syslog_2").write_text("BBB second fragment\n", encoding="utf-8")
        (workspace / "cyclic_process.sh").write_text(
            "cyclic_process\n",
            encoding="utf-8",
        )
        return workspace

    def test_skill_script_primitive_is_role_neutral(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = self._workspace(Path(temp))
            runtime = CommandRuntime(
                workspace,
                ("cyclic_process",),
                max_file_bytes=4096,
                timeout_seconds=2,
            )
            client = FakeClient(
                [],
                child_replies=["first-summary", "second-summary"],
            )

            result = run_skill_script(
                "cyclic_process.sh syslog -n 2",
                runtime,
                client,
                task_text="Кратко опиши каждый фрагмент.",
            )

            self.assertEqual(result, "syslog_out.txt")
            self.assertEqual(
                (workspace / "data" / "syslog_out.txt").read_text(encoding="utf-8"),
                "first-summary\nsecond-summary\n",
            )
            self.assertEqual(client.fork_inherit_base, [False])
            self.assertEqual(len(client.children), 1)

            child = client.children[0]
            self.assertEqual(len(child.calls), 2)
            first_prompt = child.calls[0][-1]["content"]
            second_prompt = child.calls[1][-1]["content"]
            self.assertIn("AAA first fragment", first_prompt)
            self.assertNotIn("BBB second fragment", first_prompt)
            self.assertIn("BBB second fragment", second_prompt)
            self.assertNotIn("AAA first fragment", second_prompt)
            self.assertEqual(len(child.reset_calls), 2)
            self.assertTrue(child.closed)

    def test_agent_parent_receives_only_output_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = self._workspace(root)

            client = FakeClient(
                [
                    "/work#cyclic_process.sh syslog -n 2",
                    '{"result":"syslog_out.txt"}',
                ],
                child_replies=["first-summary", "second-summary"],
            )
            prompts = PromptStore(prompt_dir, 1)
            prompts.validate()
            skill = Skill(
                name="cyclic_process",
                description="test",
                prompt=(
                    "Для выполнения skill cyclic_process используй команду:\n"
                    "/work#cyclic_process.sh"
                ),
            )
            worker = AgentWorker(
                "agent1",
                client,  # type: ignore[arg-type]
                prompts,
                workspace,
                max_steps=6,
                max_file_bytes=4096,
                command_timeout_seconds=2,
            )

            outcome = worker.start(
                "Разбери syslog по частям и выдели важное.",
                (skill,),
                method="query",
            )

            self.assertEqual(outcome.status, "OK")
            self.assertEqual(outcome.text, "syslog_out.txt")
            parent_second_call = "\n".join(
                item["content"] for item in client.calls[1]
            )
            self.assertIn("syslog_out.txt", parent_second_call)
            self.assertNotIn("first-summary", parent_second_call)
            self.assertNotIn("second-summary", parent_second_call)
            self.assertNotIn("AAA first fragment", parent_second_call)
            self.assertNotIn("BBB second fragment", parent_second_call)


if __name__ == "__main__":
    unittest.main()
