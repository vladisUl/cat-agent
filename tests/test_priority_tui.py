from __future__ import annotations

from types import SimpleNamespace
import unittest

from cat_agent.manager import AutonomousTaskCompletion, ManagerTurn
from cat_agent.system_events import SystemEvent, SystemRuntime
from litert_agent.priority_tui import (
    DEFAULT_EVENT_PRIORITY,
    MANAGER_PRIORITY,
    PriorityLiteRTTUI,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._task_steps = 0

    def begin_autonomous_task(self, event: SystemEvent):
        self.calls.append("begin")
        return SimpleNamespace(worker=SimpleNamespace(agent_id="agent1"))

    def step_autonomous_task(self, execution):
        del execution
        self._task_steps += 1
        self.calls.append(f"task-step-{self._task_steps}")
        if self._task_steps == 1:
            return None
        return AutonomousTaskCompletion(turn=ManagerTurn("silent", ""))

    def user_message(self, text: str) -> ManagerTurn:
        self.calls.append(f"user:{text}")
        return ManagerTurn("reply", "ОК")

    def autonomous_query_result(self, task_id: int, result: str) -> ManagerTurn:
        self.calls.append(f"query-result:{task_id}:{result}")
        return ManagerTurn("reply", result)

    def system_event(self, event: SystemEvent) -> ManagerTurn:
        self.calls.append(f"system:{event.source}:{event.name}")
        return ManagerTurn("silent", "")


class PriorityTUITest(unittest.TestCase):
    def setUp(self) -> None:
        self.runtime = FakeRuntime()
        bundle = SimpleNamespace(
            system_runtime=SystemRuntime(),
            runtime=self.runtime,
        )
        self.tui = PriorityLiteRTTUI(bundle)  # type: ignore[arg-type]

    def tearDown(self) -> None:
        self.tui._executor.shutdown(wait=True, cancel_futures=True)

    def test_user_manager_request_has_priority_over_background_events(self) -> None:
        self.tui._enqueue_system_event(
            SystemEvent("timer", "task:1", "", 10.0, 1)
        )
        self.tui._enqueue_system_event(
            SystemEvent("timer", "task:2", "", 20.0, 2)
        )

        self.tui._input = "температура на улице"
        self.tui._submit_input()

        first = self.tui._take_next_request()
        second = self.tui._take_next_request()
        third = self.tui._take_next_request()

        assert first is not None
        assert second is not None
        assert third is not None
        self.assertEqual(first.kind, "user")
        self.assertEqual(first.priority, MANAGER_PRIORITY)
        self.assertEqual(second.label, "timer:task:1")
        self.assertEqual(third.label, "timer:task:2")

    def test_equal_priority_events_are_fifo_by_original_event_time(self) -> None:
        self.tui.enqueue_external_event(
            SystemEvent("gpio", "second", "", 20.0),
            priority=DEFAULT_EVENT_PRIORITY,
        )
        self.tui.enqueue_external_event(
            SystemEvent("mqtt", "first", "", 10.0),
            priority=DEFAULT_EVENT_PRIORITY,
        )

        first = self.tui._take_next_request()
        second = self.tui._take_next_request()

        assert first is not None
        assert second is not None
        self.assertEqual(first.label, "mqtt:first")
        self.assertEqual(second.label, "gpio:second")

    def test_periodic_timer_keeps_first_pending_activation_and_drops_duplicates(self) -> None:
        self.tui._enqueue_system_event(
            SystemEvent("timer", "task:1", "", 10.0, 1)
        )
        self.tui._enqueue_system_event(
            SystemEvent("timer", "task:1", "", 20.0, 1)
        )
        self.tui._enqueue_system_event(
            SystemEvent("timer", "task:1", "", 30.0, 1)
        )

        self.assertEqual(len(self.tui._pending), 1)
        request = self.tui._take_next_request()
        assert request is not None
        self.assertEqual(request.queued_at, 10.0)
        assert isinstance(request.payload, SystemEvent)
        self.assertEqual(request.payload.created_monotonic, 10.0)

    def test_external_callback_has_configurable_priority_and_coalescing(self) -> None:
        callback = self.tui.make_event_callback(priority=25, coalesce=True)
        callback(SystemEvent("gpio", "water", "close", 10.0))
        callback(SystemEvent("gpio", "water", "close", 20.0))
        self.tui.enqueue_external_event(
            SystemEvent("mqtt", "status", "read", 5.0),
            priority=DEFAULT_EVENT_PRIORITY,
        )

        self.assertEqual(len(self.tui._pending), 2)
        first = self.tui._take_next_request()
        second = self.tui._take_next_request()

        assert first is not None
        assert second is not None
        self.assertEqual(first.label, "gpio:water")
        self.assertEqual(first.priority, 25)
        self.assertEqual(first.queued_at, 10.0)
        self.assertEqual(second.label, "mqtt:status")

    def test_external_event_cannot_outrank_manager(self) -> None:
        with self.assertRaises(ValueError):
            self.tui.make_event_callback(priority=MANAGER_PRIORITY)

    def test_user_runs_between_tt_steps_of_one_background_activation(self) -> None:
        self.tui.enqueue_external_event(
            SystemEvent("gpio", "task:1", "", 10.0, 1),
            priority=DEFAULT_EVENT_PRIORITY,
            coalesce=True,
        )

        self.tui._start_next()
        assert self.tui._active_future is not None
        self.tui._active_future.result(timeout=1)
        self.tui._poll_future()

        self.assertIsNotNone(self.tui._background_execution)
        self.assertEqual(
            self.runtime.calls,
            ["begin", "task-step-1"],
        )

        self.tui._input = "температура на улице"
        self.tui._submit_input()
        self.tui._start_next()
        assert self.tui._active_future is not None
        self.tui._active_future.result(timeout=1)
        self.tui._poll_future()

        self.assertEqual(
            self.runtime.calls,
            ["begin", "task-step-1", "user:температура на улице"],
        )
        self.assertIsNotNone(self.tui._background_execution)

        self.tui._start_next()
        assert self.tui._active_future is not None
        self.tui._active_future.result(timeout=1)
        self.tui._poll_future()

        self.assertEqual(
            self.runtime.calls,
            [
                "begin",
                "task-step-1",
                "user:температура на улице",
                "task-step-2",
            ],
        )
        self.assertIsNone(self.tui._background_execution)

    def test_duplicate_event_is_coalesced_while_activation_is_tt_paused(self) -> None:
        event = SystemEvent("gpio", "task:1", "", 10.0, 1)
        self.tui.enqueue_external_event(
            event,
            priority=DEFAULT_EVENT_PRIORITY,
            coalesce=True,
        )
        self.tui._start_next()
        assert self.tui._active_future is not None
        self.tui._active_future.result(timeout=1)
        self.tui._poll_future()

        self.tui.enqueue_external_event(
            SystemEvent("gpio", "task:1", "", 20.0, 1),
            priority=DEFAULT_EVENT_PRIORITY,
            coalesce=True,
        )
        self.assertEqual(len(self.tui._pending), 0)


if __name__ == "__main__":
    unittest.main()
