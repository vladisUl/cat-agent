from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from orchestration.agent import AgentWorker
from orchestration.assistant_manager import AssistantManagerRuntime
from orchestration.event_store import EventStore
from orchestration.manager import AutonomousTaskExecution
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

    def test_manager_base_appends_mqtt_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runtime, _client = self._runtime(root, ["REPLY\nunused"])
            system_prompt = (root / "prompts" / "sys_prompt_manager.txt").read_text(
                encoding="utf-8"
            ).strip()
            mqtt_catalog = (root / "prompts" / "mqtt.txt").read_text(
                encoding="utf-8"
            ).strip()
            expected = f"{system_prompt}\n{mqtt_catalog}"
            self.assertEqual(runtime.messages[0]["content"].strip(), expected)
            self.assertNotIn("[MANAGER_TOOLS]", runtime.messages[0]["content"])
            self.assertNotIn("[AGENT_EXECUTION_PROTOCOL]", runtime.messages[0]["content"])
            self.assertIn("zigbee2mqtt/dvigen_verh", runtime.messages[0]["content"])

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

    def test_periodic_query_uses_task_text_as_saved_description(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, _client = self._runtime(Path(temp), ["REPLY\nunused"])

            text = "Проверять температуру на улице и сообщать мне"
            result = runtime._execute_task_command(
                ["query_timer.sh", "60", "mqtt", text]
            )

            self.assertEqual(result, "SYSTEM_OK\nTASK 1 created and started")
            task = runtime.system_runtime.task_store.require(1)  # type: ignore[union-attr]
            self.assertEqual(task.method, "query")
            self.assertEqual(task.text, text)
            self.assertEqual(task.description, text)
            self.assertEqual(task.skills, ("mqtt",))
            self.assertEqual(task.timer_period_seconds, 60.0)

    def test_external_mqtt_query_defers_tool_result_until_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = self._runtime(
                Path(temp),
                [
                    "/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
                    '{"result":"В коридоре обнаружено движение"}',
                ],
            )

            text = "Сообщить мне, когда в коридоре появится движение"
            result = runtime._execute_task_command(
                ["query_timer.sh", "-1", "mqtt", text]
            )

            self.assertEqual(result, "SYSTEM_OK\nTASK 1 created and started")
            task = runtime.system_runtime.task_store.require(1)  # type: ignore[union-attr]
            self.assertEqual(task.method, "query")
            self.assertEqual(task.text, text)
            self.assertIsNone(task.timer_period_seconds)

            binding = runtime.event_store.resolve("mqtt", "task_mqtt1")
            self.assertIsNotNone(binding)
            assert binding is not None
            self.assertEqual(binding.task_id, 1)
            self.assertEqual(binding.topic, "zigbee2mqtt/dvigen_verh")
            self.assertEqual(binding.field, "occupancy")
            self.assertEqual(binding.value_type, "boolean")
            self.assertEqual(binding.values, ("true",))
            self.assertEqual(
                binding.command,
                "mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
            )

            event = runtime.external_event("mqtt", "task_mqtt1", value="true")
            self.assertIsNotNone(event)
            assert event is not None
            self.assertEqual(event.task_id, 1)
            self.assertEqual(event.task, "true")

            execution = runtime.begin_autonomous_task(event)
            self.assertIsInstance(execution, AutonomousTaskExecution)
            assert isinstance(execution, AutonomousTaskExecution)
            completion = runtime.step_autonomous_task(execution)
            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.query_task_id, 1)
            self.assertEqual(completion.query_result, "В коридоре обнаружено движение")

            resumed_messages = client.calls[-1]
            self.assertEqual(resumed_messages[-3]["content"].strip(), text)
            self.assertEqual(
                resumed_messages[-2]["content"],
                "/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
            )
            self.assertEqual(resumed_messages[-1]["content"], "true")

    def test_invalid_negative_period_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime, _client = self._runtime(Path(temp), ["REPLY\nunused"])
            result = runtime._execute_task_command(
                ["task_timer.sh", "-0.5", "shell", "do bad"]
            )
            self.assertEqual(
                result,
                "SYSTEM_ERROR\nperiod_seconds must be -1, 0 or > 0",
            )


if __name__ == "__main__":
    unittest.main()
