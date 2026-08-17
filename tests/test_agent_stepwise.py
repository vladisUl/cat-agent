from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentState, AgentWorker
from orchestration.model_client import ChatResponse
from orchestration.prompt_store import PromptStore
from orchestration.skills import SkillBase


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


class AgentStepwiseTest(unittest.TestCase):
    def test_one_step_is_one_model_tt_and_same_activation_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()

            client = FakeClient(["echo ok", '{"done":true}'])
            prompts = PromptStore(prompt_dir, 1)
            prompts.validate()
            skills = SkillBase(prompt_dir / "prompt_base.txt").require(("shell",))
            worker = AgentWorker(
                "agent1",
                client,  # type: ignore[arg-type]
                prompts,
                workspace,
                max_steps=6,
                max_file_bytes=1024,
                command_timeout_seconds=2,
            )

            worker.begin("Выполнить команду.", skills, method="task")
            self.assertEqual(worker.state, AgentState.RUNNING)
            self.assertEqual(len(client.calls), 0)

            first = worker.step()
            self.assertIsNone(first)
            self.assertEqual(worker.state, AgentState.RUNNING)
            self.assertEqual(len(client.calls), 1)

            second = worker.step()
            assert second is not None
            self.assertEqual(second.status, "OK")
            self.assertEqual(second.steps, 2)
            self.assertEqual(worker.state, AgentState.FREE)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(client.calls[1][-1]["role"], "user")
            self.assertIn("ok", client.calls[1][-1]["content"])
            self.assertEqual(len(client.reset_calls), 1)


if __name__ == "__main__":
    unittest.main()
