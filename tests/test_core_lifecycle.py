from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import tempfile
import unittest

from agent_core.core_scheduler import CoreScheduler, QueueFull, _PriorityRequest
from agent_core.core_server import CoreServer
from agent_core.voice_scheduler import VoiceCoreScheduler
from orchestration.manager import ManagerTurn, AutonomousTaskExecution
from orchestration.system_events import SystemEvent
import test_assistant_manager as fixtures


class Recipient:
    def __init__(self, session_id):
        self.session_id = session_id
        self.client_name = session_id
        self.human_owner = False
        self.messages = []

    def send(self, item):
        self.messages.append(item)
        return True


def finish(scheduler, turn):
    future = Future()
    future.set_result(turn)
    scheduler._finish_regular_request(future)


class CoreLifecycleTest(unittest.TestCase):
    def server(self):
        server = CoreServer(SimpleNamespace(runtime=SimpleNamespace(human_session_released=lambda: None)))
        self.addCleanup(server.scheduler._executor.shutdown, wait=True)
        return server

    def test_silent_voice_completes_and_next_request_is_accepted(self):
        server = self.server()
        client = Recipient("voice-session")
        server._submit_voice_turn(client, "one", "voice")
        server.scheduler._active_request = server.scheduler._take_next_request()
        finish(server.scheduler, ManagerTurn("silent", ""))
        self.assertEqual(client.messages[-1]["type"], "completed")
        self.assertEqual(client.messages[-1]["kind"], "silent")
        self.assertIsNone(server._voice_request_id)
        server._submit_voice_turn(client, "two", "voice")
        self.assertEqual(client.messages[-1]["type"], "voice_accepted")

    def test_old_reply_cannot_reach_new_owner(self):
        server = self.server()
        old, new = Recipient("old"), Recipient("new")
        server._human = new
        item = _PriorityRequest("user", "user", "old question", 0, 0, session_id=old.session_id)
        server._deliver_completed(item, ManagerTurn("reply", "old answer"))
        self.assertEqual(new.messages, [])

    def test_release_cancels_only_own_requests_and_keeps_voice(self):
        server = self.server()
        s = server.scheduler
        s.submit_user("old", session_id="old")
        s.submit_user("new", session_id="new")
        s.submit_voice("voice", session_id="voice")
        s.release_human_session("old")
        states = {r.session_id: r.cancelled for r in s._pending}
        self.assertEqual(states, {"old": True, "new": False, "voice": False})

    def test_queue_admission_is_bounded(self):
        s = CoreScheduler(SimpleNamespace(), max_pending=2)
        self.addCleanup(s._executor.shutdown, wait=True)
        s.submit_user("a")
        s.submit_user("b")
        with self.assertRaises(QueueFull):
            s.submit_user("c")

    def test_stopped_queued_mqtt_is_cancelled(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, _ = fixtures.AssistantManagerTest()._runtime(Path(temp), ["/work#mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy"])
            runtime._execute_task_command(["task_timer.sh", "-1", "mqtt", "сообщить о движении"])
            event = runtime.external_event("mqtt", "task_mqtt1", value="true")
            runtime._execute_timer_command(["timer.sh", "stop", "1"])
            self.assertEqual(runtime.begin_autonomous_task(event).turn.kind, "cancelled")
            runtime._execute_timer_command(["timer.sh", "start", "1"])
            self.assertEqual(runtime.begin_autonomous_task(event).turn.kind, "cancelled")

    def test_delete_recreate_does_not_reuse_event_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, _ = fixtures.AssistantManagerTest()._runtime(Path(temp), [])
            task = runtime.system_runtime.create_periodic_task("old", "old", ("shell",), 60, method="query")
            event = SystemEvent("timer", "task:1", "", 0, task.task_id, task.generation)
            runtime.system_runtime.delete_task(1)
            new = runtime.system_runtime.create_periodic_task("new", "new", ("shell",), 60, method="query")
            self.assertEqual(new.task_id, task.task_id)
            self.assertNotEqual(new.generation, task.generation)
            self.assertEqual(runtime.begin_autonomous_task(event).turn.kind, "cancelled")

    def test_stop_between_steps_releases_worker_without_next_command(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = fixtures.AssistantManagerTest()._runtime(Path(temp), ["/work#printf first", "/work#touch forbidden"])
            task = runtime.system_runtime.create_periodic_task("test", "test", ("shell",), 60, method="query")
            execution = runtime.begin_autonomous_task(SystemEvent("timer", "task:1", "", 0, task.task_id, task.generation))
            self.assertIsInstance(execution, AutonomousTaskExecution)
            runtime.step_autonomous_task(execution)
            runtime.system_runtime.stop_task(task.task_id)
            completion = runtime.step_autonomous_task(execution)
            self.assertEqual(completion.turn.kind, "cancelled")
            self.assertEqual(execution.worker.state.value, "FREE")
            self.assertEqual(len(client.calls), 1)

    def test_mqtt_keeps_payload_when_another_task_is_paused(self):
        seen = []
        workers = {}
        def begin(event):
            seen.append((event.name, event.task))
            worker = SimpleNamespace(state=SimpleNamespace(value="FREE"))
            execution = SimpleNamespace(worker=worker, event=event, count=0)
            return execution
        def step(execution):
            execution.count += 1
            if execution.count == 1:
                return None
            from orchestration.manager import AutonomousTaskCompletion
            return AutonomousTaskCompletion(turn=ManagerTurn("silent", ""))
        runtime = SimpleNamespace(begin_autonomous_task=begin, step_autonomous_task=step)
        s = CoreScheduler(SimpleNamespace(runtime=runtime))
        self.addCleanup(s._executor.shutdown, wait=True)
        s.enqueue_external_event(SystemEvent("timer", "task:1", "", 1, 1), priority=100)
        s._start_next(); s._active_future.result(timeout=1); s._poll_future()
        s.enqueue_external_event(SystemEvent("mqtt", "task_mqtt2", "true", 2, 2), priority=10)
        s._start_next(); s._active_future.result(timeout=1); s._poll_future()
        self.assertEqual(seen, [("task:1", ""), ("task_mqtt2", "true")])

    def test_voice_preempts_manager_between_steps_with_distinct_context(self):
        calls = []
        class Context:
            _chat_mode = False
            def __init__(self, label):
                self.label = label
                self.client = SimpleNamespace(set_event_handler=lambda h: None, close=lambda: None)
            def user_message_steps(self, text):
                calls.append((self.label, text, 1))
                yield
                calls.append((self.label, text, 2))
                return ManagerTurn("reply", text)
        runtime = SimpleNamespace(fork_context=lambda key: Context(key))
        s = VoiceCoreScheduler(SimpleNamespace(runtime=runtime))
        self.addCleanup(s._executor.shutdown, wait=True)
        s.submit_user("terminal", session_id="human")
        s._start_next(); s._active_future.result(timeout=1); s._poll_future()
        s.submit_voice("spoken", session_id="microphone")
        for _ in range(3):
            s._start_next(); s._active_future.result(timeout=1); s._poll_future()
        self.assertEqual(calls, [("human:human", "terminal", 1), ("voice", "spoken", 1), ("voice", "spoken", 2), ("human:human", "terminal", 2)])

class RealManagerSchedulingTest(unittest.TestCase):
    def test_warmed_human_context_reused_without_history_or_new_fork(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = fixtures.AssistantManagerTest()._runtime(Path(temp), [])
            child = fixtures.FakeClient([])
            child.set_event_handler = Mock()
            child.close = Mock()
            client.fork = Mock(return_value=child)
            s = CoreScheduler(SimpleNamespace(runtime=runtime))
            self.addCleanup(s._executor.shutdown, wait=True)
            s._prepare_human_context()
            first = _PriorityRequest("user", "user", "one", 0, 0, session_id="old")
            context = s._context(first)
            context.messages.append({"role": "user", "content": "private history"})
            context._chat_mode = True
            context._direct_repeated = {"private command": 1}
            context._direct_runtime._uncertain_commands = {"private command"}
            s._release_context("old")
            second = _PriorityRequest("user", "user", "two", 0, 0, session_id="new")
            self.assertIs(s._context(second), context)
            self.assertEqual(client.fork.call_count, 1)
            self.assertEqual(child.reset_calls, [runtime._base_messages])
            self.assertEqual(context.messages, runtime._base_messages)
            self.assertFalse(context._chat_mode)
            self.assertEqual(context._direct_repeated, {})
            self.assertEqual(context._direct_runtime._uncertain_commands, set())
            child.close.assert_not_called()
            s._stop.set()
            s._close_all_contexts()
            child.close.assert_called_once()

    def test_failed_kv_reset_discards_context(self):
        s = CoreScheduler(SimpleNamespace(runtime=SimpleNamespace()))
        self.addCleanup(s._executor.shutdown, wait=True)
        context = SimpleNamespace(
            reset_for_new_session=Mock(side_effect=RuntimeError("KV reset failed")),
            client=SimpleNamespace(close=Mock()))
        s._contexts["human:old"] = context
        with self.assertLogs("agent_core.core_scheduler", level="ERROR"):
            s._close_context("human:old")
        self.assertIsNone(s._spare_human_context)
        context.client.close.assert_called_once()

    def test_release_waits_for_old_requests_before_recycling(self):
        s = CoreScheduler(SimpleNamespace(runtime=SimpleNamespace()))
        self.addCleanup(s._executor.shutdown, wait=True)
        context = SimpleNamespace(reset_for_new_session=Mock(), client=SimpleNamespace(close=Mock()))
        s._contexts["human:old"] = context
        s.submit_user("unfinished", session_id="old")
        s._release_context("old")
        context.reset_for_new_session.assert_not_called()
        self.assertIsNone(s._spare_human_context)
        s._pending.clear()
        s._release_context("old")
        context.reset_for_new_session.assert_called_once()
        self.assertIs(s._spare_human_context, context)

    def test_real_manager_tool_flow_resumes_after_voice(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = fixtures.AssistantManagerTest()._runtime(Path(temp), [])
            profiles = iter([["/work#printf terminal-result", "REPLY\nterminal-result"], ["REPLY\nvoice-result"]])
            def fork(label):
                child = fixtures.FakeClient(next(profiles))
                child.set_event_handler = lambda h: None
                child.close = lambda: None
                return child
            client.fork = fork
            delivered = []
            s = VoiceCoreScheduler(SimpleNamespace(runtime=runtime), on_completed=lambda r,t: delivered.append((r.label,t.text)))
            try:
                s.submit_user("terminal", session_id="human")
                s._start_next(); s._active_future.result(timeout=1); s._poll_future()
                s.submit_voice("voice", session_id="microphone")
                for _ in range(20):
                    if not s._pending:
                        break
                    s._start_next()
                    self.assertIsNotNone(s._active_future)
                    s._active_future.result(timeout=1); s._poll_future()
                self.assertFalse(s._pending)
                self.assertEqual(delivered, [("voice", "voice-result"), ("user", "terminal-result")])
                self.assertIsNot(s._contexts['voice'].client, s._contexts['human:human'].client)
            finally:
                s._executor.shutdown(wait=True)
