"""Stage-3B: no nested inference, queue lifecycle and outcome delivery tests."""
from __future__ import annotations

import json
import unittest

from orchestration_tools.agents import AgentOutcome, AgentState
from orchestration_tools.delegation import (
    DelegationBroker, MAX_DELEGATIONS_PER_TURN, create_delegate_tool,
)
from orchestration_tools.manager import NativeManager


class FakeChat:
    base_kv_tokens = 300
    def __init__(self, tool, events):
        self.tool = tool
        self.events = events
        self.fail = False
        self.on_first = None
        self.closed = False
        self.prompts = []

    def ask(self, question):
        self.prompts.append(question)
        if "CAT_AGENT_DELEGATION_RESULTS" in question:
            self.events.append("manager_follows_up")
            payload = json.loads(question.split("\n")[-1])
            results = payload["results"]
            return "Итог: " + "; ".join(str(item["result"]) for item in results)
        if question.startswith("Поручи"):
            self.events.append("manager_inference")
            self.tool("agent_1", "Выполни shell pwd")
            if self.on_first:
                self.on_first()
            if self.fail:
                raise RuntimeError("native generation failed")
            return "Запрос поставлен в очередь"
        self.events.append("manager_answer")
        return "Обычный ответ"

    def new_dialogue(self):
        self.events.append("manager_reset")

    def close(self):
        self.closed = True


class FakePool:
    def __init__(self, events):
        self.events = events
        self.calls = []
        self.fail = False
        self.closed = False

    def run(self, task, agent_id):
        self.events.append("agent_inference")
        self.calls.append((agent_id, task))
        status = AgentState.FAILED if self.fail else AgentState.FREE
        return AgentOutcome(agent_id, status, "ошибка" if self.fail else "/opt/model", 350)

    def close(self):
        self.closed = True

    def status(self):
        return [{"agent_id": "agent_1", "state": "FREE", "base_kv_tokens": 300}]


class Engine:
    def __init__(self):
        self.closed = False
    def close(self):
        self.closed = True


class DelegationTests(unittest.TestCase):
    def make_manager(self):
        events = []
        broker = DelegationBroker(("agent_1",))
        tool = create_delegate_tool(broker)
        chat = FakeChat(tool, events)
        pool = FakePool(events)
        engine = Engine()
        manager = NativeManager(chat, engine, model_name="gemma", agents=pool,
                                delegations=broker)
        broker.enabled = True
        return manager, broker, tool, events, chat, pool, engine

    def test_no_agent_called_inside_native_tool_callback(self):
        manager, broker, tool, events, chat, pool, _ = self.make_manager()
        reply = manager.respond("client", "Поручи агенту проверить pwd")
        self.assertEqual(reply.text, "Итог: /opt/model")
        self.assertEqual(events, [
            "manager_inference", "agent_inference", "manager_follows_up",
        ])
        self.assertEqual(pool.calls, [("agent_1", "Выполни shell pwd")])
        self.assertFalse(broker._turn_active)
        self.assertEqual(broker.take_pending(), ())
        self.assertEqual(manager.respond("client", "Привет").text, "Обычный ответ")
        manager.close()

    def test_agent_failure_is_reported_once_without_retry(self):
        manager, broker, _, events, _, pool, _ = self.make_manager()
        pool.fail = True
        answer = manager.respond("client", "Поручи задачу")
        self.assertIn("ошибка", answer.text)
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(events.count("manager_follows_up"), 1)
        manager.close()

    def test_cancelled_after_initial_manager_turn_skips_side_effects(self):
        manager, broker, _, events, chat, pool, _ = self.make_manager()
        state = {"active": True}
        chat.on_first = lambda: state.update(active=False)
        answer = manager.respond_guarded(
            "client", "Поручи агенту", lambda: state["active"],
        )
        self.assertEqual(answer.kind, "silent")
        self.assertEqual(pool.calls, [])
        self.assertEqual(events, ["manager_inference"])
        self.assertFalse(broker._turn_active)
        manager.close()

    def test_raised_native_error_discards_pending_work(self):
        manager, broker, _, events, chat, pool, _ = self.make_manager()
        chat.fail = True
        with self.assertRaisesRegex(RuntimeError, "native generation failed"):
            manager.respond("client", "Поручи задачу")
        self.assertEqual(pool.calls, [])
        self.assertFalse(broker._turn_active)
        self.assertEqual(broker.take_pending(), ())
        chat.fail = False
        self.assertEqual(manager.respond("client", "Привет").text, "Обычный ответ")
        manager.close()

    def test_broker_guards_and_fixed_budget(self):
        broker = DelegationBroker(("agent_1",))
        tool = create_delegate_tool(broker)
        self.assertEqual(tool.__name__, "delegate_task")
        self.assertEqual(tool("agent_1", "task")["error"],
                         "delegation_disabled_during_warm")
        broker.enabled = True
        self.assertEqual(tool("agent_1", "task")["error"], "no_active_manager_turn")
        broker.begin_turn()
        self.assertEqual(tool("agent_2", "task")["error"], "unknown_agent")
        self.assertEqual(tool("agent_1", "")["error"], "invalid_task")
        self.assertEqual(tool("agent_1", chr(0xDCD1))["error"], "invalid_unicode")
        self.assertEqual(tool("agent_1", "x" * 4097)["error"], "invalid_task")
        for i in range(MAX_DELEGATIONS_PER_TURN):
            self.assertEqual(tool("agent_1", f"task {i}")["status"], "queued")
        self.assertEqual(tool("agent_1", "one more")["error"], "delegation_limit_reached")
        self.assertEqual(len(broker.take_pending()), MAX_DELEGATIONS_PER_TURN)
        self.assertEqual(broker.take_pending(), ())
        with self.assertRaisesRegex(RuntimeError, "Nested"):
            broker.begin_turn()
        broker.finish_turn()
        broker.begin_turn()
        self.assertEqual(tool("agent_1", "new")["status"], "queued")
        broker.finish_turn()

    def test_query_manager_is_independent_of_interactive_dialogue(self):
        manager, _, _, events, chat, pool, engine = self.make_manager()
        class QueryChat:
            def __init__(self):
                self.prompts = []
                self.resets = 0
                self.closed = False
            def ask(self, question):
                self.prompts.append(question)
                return "Итог задания: каталог /opt/model"
            def new_dialogue(self):
                self.resets += 1
            def close(self):
                self.closed = True
        query = QueryChat()
        manager.query_chat = query
        self.assertEqual(manager.respond("human-1", "Привет").kind, "reply")
        prior_prompts = list(chat.prompts)
        outcome = manager.process_task_query(3, "/opt/model")
        self.assertEqual(outcome.kind, "reply")
        self.assertIn("Итог задания", outcome.text)
        self.assertEqual(chat.prompts, prior_prompts)
        self.assertEqual(query.resets, 1)
        self.assertEqual(len(query.prompts), 1)
        self.assertIn("SYSTEM_QUERY_RESULT TASK 3", query.prompts[0])
        self.assertIn('"task_id":3', query.prompts[0])
        manager.close()
        self.assertTrue(query.closed)
        self.assertTrue(chat.closed)
        self.assertTrue(engine.closed)

    def test_query_manager_reset_even_if_inference_fails(self):
        manager, _, _, _, _, _, _ = self.make_manager()
        class FailedQuery:
            def __init__(self):
                self.resets = 0
            def ask(self, question):
                raise RuntimeError("query manager failed")
            def new_dialogue(self):
                self.resets += 1
            def close(self):
                pass
        chat = FailedQuery()
        manager.query_chat = chat
        with self.assertRaisesRegex(RuntimeError, "query manager failed"):
            manager.process_task_query(1, "data")
        self.assertEqual(chat.resets, 1)
        manager.close()

    def test_builder_closes_model_and_pool(self):
        manager, _, _, _, chat, pool, engine = self.make_manager()
        manager.close()
        self.assertTrue(chat.closed)
        self.assertTrue(pool.closed)
        self.assertTrue(engine.closed)


if __name__ == "__main__":
    unittest.main()
