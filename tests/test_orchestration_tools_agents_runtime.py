"""Engine sharing and per-agent native tool registrations, without real LiteRT."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from orchestration_tools.litert_runtime import build_native_manager


class FakeConversation:
    def __init__(self, options, base_kv=0):
        self.options = options
        self.token_count = base_kv
        self.closed = False
        self.history = []

    def send_message(self, question):
        self.history.append(question)
        self.token_count += 20
        return {"role": "assistant", "content": "OK" if question == "warm" else "DONE:" + question}

    def clone(self):
        result = FakeConversation(self.options, self.token_count)
        result.history = list(self.history)
        return result

    def close(self):
        self.closed = True


class FakeEngine:
    instances = []
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.created = []
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def create_conversation(self, **kwargs):
        self.created.append(kwargs)
        return FakeConversation(kwargs)

    def close(self):
        self.closed = True


class RuntimeAgentTests(unittest.TestCase):
    def test_manager_and_agents_share_one_engine_but_have_distinct_prefaces(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.litertlm"
            model.touch()
            manager_prompt = root / "manager.md"
            manager_prompt.write_text("Я менеджер", encoding="utf-8")
            agent_prompt = root / "agent.md"
            agent_prompt.write_text("Я агент", encoding="utf-8")
            (root / "query_manager.md").write_text(
                "TASK QUERY: Сформулируй результат автономной задачи.",
                encoding="utf-8",
            )
            config = root / "agents.json"
            config.write_text(json.dumps({
                "agents": [{"id": "agent_1", "prompt": "agent.md", "tools": []}]
            }), encoding="utf-8")
            fake = ModuleType("litert_lm")
            fake.Engine = FakeEngine
            fake.Backend = SimpleNamespace(CPU=lambda **kwargs: "cpu")
            fake.SamplerConfig = lambda **kwargs: "sampler"
            fake.ThinkingConfig = lambda **kwargs: "thinking"
            FakeEngine.instances.clear()
            with patch.dict(sys.modules, {"litert_lm": fake}):
                manager = build_native_manager(
                    model, manager_prompt,
                    agent_config=config,
                    enable_mqtt=False,
                    enable_shell=False,
                )
            engine = FakeEngine.instances[0]
            self.assertEqual(len(FakeEngine.instances), 1)
            self.assertEqual(len(engine.created), 3)
            self.assertTrue(engine.created[0]["messages"][0]["content"].startswith("Я менеджер"))
            self.assertIn("agent_1", engine.created[0]["messages"][0]["content"])
            self.assertEqual(engine.created[1]["messages"][0]["content"], "Я агент")
            self.assertEqual(engine.created[1]["tools"], [])
            self.assertEqual(engine.created[2]["tools"], [])
            self.assertIn("TASK QUERY", engine.created[2]["messages"][0]["content"])
            self.assertEqual(len(engine.created[0]["tools"]), 1)
            self.assertEqual(engine.created[0]["tools"][0].__name__, "delegate_task")
            self.assertEqual(
                engine.created[0]["tools"][0]("agent_1", "проверить"),
                {"ok": False, "error": "no_active_manager_turn"},
            )
            self.assertEqual(manager.base_kv_tokens, 20)
            self.assertEqual(manager.agent_status()[0]["base_kv_tokens"], 20)
            self.assertEqual(manager.respond("test", "Привет").text, "DONE:Привет")
            with self.assertLogs("orchestration_tools.manager", level="INFO") as events:
                self.assertIn("SYSTEM_QUERY_RESULT TASK 1",
                              manager.process_task_query(1, "/opt/model").text)
            journal = "\\n".join(events.output)
            self.assertIn("SYSTEM -> QUERY_MANAGER", journal)
            self.assertIn("QUERY_MANAGER -> SYSTEM", journal)
            self.assertEqual(manager.run_agent("agent_1", "работай").text, "DONE:работай")
            self.assertEqual(manager.run_agent("agent_1", "еще").text, "DONE:еще")
            self.assertEqual(manager.agent_status()[0]["state"], "FREE")
            manager.close()
            self.assertTrue(engine.closed)

    def test_config_cannot_grant_disabled_or_unknown_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.litertlm"
            model.touch()
            prompt = root / "manager.md"
            prompt.write_text("manager", encoding="utf-8")
            agent_prompt = root / "agent.md"
            agent_prompt.write_text("agent", encoding="utf-8")
            config = root / "agents.json"
            config.write_text(json.dumps({"agents": [
                {"id": "agent_1", "prompt": "agent.md", "tools": ["shell"]}
            ]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "disabled tools"):
                build_native_manager(model, prompt, agent_config=config,
                                     enable_shell=False, enable_mqtt=False)


if __name__ == "__main__":
    unittest.main()
