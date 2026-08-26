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
from orchestration.system_events import SystemRuntime, TaskActivation
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


class QueryOutcomeTest(unittest.TestCase):
    def _runtime(
        self,
        root: Path,
        replies: list[str],
    ) -> AssistantManagerRuntime:
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
        return AssistantManagerRuntime(
            client,  # type: ignore[arg-type]
            SkillBase(prompt_dir / "prompt_base.txt"),
            store,
            AgentPool([worker]),
            SystemRuntime(TaskStore(root / "task.txt")),
            max_steps=8,
            event_store=EventStore(root / "events.json"),
        )

    def test_external_query_done_is_silent_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = self._runtime(
                Path(temp),
                [
                    "/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
                    "/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
                    '{"done":true}',
                ],
            )

            created = runtime._execute_task_command(
                [
                    "query_timer.sh",
                    "-1",
                    "mqtt",
                    "При регистрации движения в коридоре сообщать мне",
                ]
            )
            self.assertEqual(created, "SYSTEM_OK\nTASK 1 created and started")

            event = runtime.external_event("mqtt", "task_mqtt1", value="false")
            self.assertIsNotNone(event)
            assert event is not None

            execution = runtime.begin_autonomous_task(event)
            self.assertIsInstance(execution, AutonomousTaskExecution)
            assert isinstance(execution, AutonomousTaskExecution)

            self.assertIsNone(runtime.step_autonomous_task(execution))
            completion = runtime.step_autonomous_task(execution)
            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertIsNotNone(completion.turn)
            assert completion.turn is not None
            self.assertEqual(completion.turn.kind, "silent")
            self.assertIsNone(completion.query_task_id)
            self.assertEqual(completion.query_result, "")

    def test_external_query_need_becomes_query_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = self._runtime(
                Path(temp),
                [
                    "/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
                    "/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
                    '{"need":"датчик недоступен"}',
                ],
            )

            runtime._execute_task_command(
                [
                    "query_timer.sh",
                    "-1",
                    "mqtt",
                    "При регистрации движения в коридоре сообщать мне",
                ]
            )
            event = runtime.external_event("mqtt", "task_mqtt1", value="true")
            self.assertIsNotNone(event)
            assert event is not None

            execution = runtime.begin_autonomous_task(event)
            self.assertIsInstance(execution, AutonomousTaskExecution)
            assert isinstance(execution, AutonomousTaskExecution)

            self.assertIsNone(runtime.step_autonomous_task(execution))
            completion = runtime.step_autonomous_task(execution)
            self.assertIsNotNone(completion)
            assert completion is not None
            self.assertEqual(completion.query_task_id, 1)
            self.assertEqual(
                completion.query_result,
                "Ошибка запроса TASK 1: датчик недоступен",
            )

    def test_periodic_query_done_returns_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            runtime = self._runtime(Path(temp), ['{"done":true}'])
            task = runtime.system_runtime.create_periodic_task(
                "Проверять условие",
                "Проверять условие и сообщать только при его выполнении",
                ("shell",),
                60,
                method="query",
            )
            activation = TaskActivation(
                source="timer",
                name=f"task_{task.task_id}",
                task=task,
                created_monotonic=0.0,
            )

            self.assertIsNone(runtime._run_task_activation(activation))


if __name__ == "__main__":
    unittest.main()
