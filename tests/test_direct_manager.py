from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentWorker
from orchestration.direct_manager import DirectSessionManagerRuntime
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


class DirectManagerTest(unittest.TestCase):
    def test_sam_uses_separate_plain_text_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prompt_dir = root / "prompts"
            shutil.copytree(Path(__file__).resolve().parents[1] / "prompts", prompt_dir)
            workspace = root / "workspace"
            workspace.mkdir()

            manager_client = FakeClient([])
            direct_client = FakeClient([
                "/work#printf self-ok",
                "self-ok",
            ])
            store = PromptStore(prompt_dir, 1)
            store.validate()
            worker = AgentWorker(
                "agent1",
                manager_client,  # type: ignore[arg-type]
                store,
                workspace,
                max_steps=6,
                max_file_bytes=4096,
                command_timeout_seconds=2,
            )
            runtime = DirectSessionManagerRuntime(
                manager_client,  # type: ignore[arg-type]
                direct_client,  # type: ignore[arg-type]
                SkillBase(prompt_dir / "prompt_base.txt"),
                store,
                AgentPool([worker]),
                SystemRuntime(TaskStore(root / "task.txt")),
                max_steps=8,
            )

            turn = runtime.user_message("СаМ выведи self-ok")

            self.assertEqual(turn.kind, "reply")
            self.assertEqual(turn.text, "self-ok")
            self.assertEqual(manager_client.calls, [])
            self.assertEqual(len(direct_client.calls), 2)
            first = direct_client.calls[0]
            self.assertEqual(first[0]["role"], "system")
            self.assertIn("прямого режима САМ", first[0]["content"])
            self.assertIn("/work#<команда>", first[0]["content"])
            self.assertIn('"name": "mqtt"', first[0]["content"])
            self.assertEqual(first[-1]["role"], "user")
            self.assertEqual(first[-1]["content"], "выведи self-ok")
            self.assertNotIn("СаМ", first[-1]["content"])
            self.assertNotIn('"task"', first[-1]["content"])
            self.assertIn("self-ok", direct_client.calls[1][-1]["content"])


if __name__ == "__main__":
    unittest.main()
