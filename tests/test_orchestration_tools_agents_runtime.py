"""Engine sharing and per-agent native tool registrations, without real LiteRT."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from litert_agent_tools.mqtt_tools import create_mqtt_tools
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
            self.assertEqual(len(engine.created), 2)
            self.assertTrue(engine.created[0]["messages"][0]["content"].startswith("Я менеджер"))
            self.assertIn("agent_1", engine.created[0]["messages"][0]["content"])
            self.assertTrue(engine.created[1]["messages"][0]["content"].startswith("Я агент"))
            self.assertEqual(engine.created[1]["tools"], [])
            self.assertEqual(len(engine.created[0]["tools"]), 1)
            self.assertEqual(engine.created[0]["tools"][0].__name__, "delegate_task")
            self.assertEqual(
                engine.created[0]["tools"][0]("agent_1", "проверить"),
                {"ok": False, "error": "no_active_manager_turn"},
            )
            self.assertEqual(manager.base_kv_tokens, 20)
            self.assertEqual(manager.agent_status()[0]["base_kv_tokens"], 20)
            self.assertEqual(manager.respond("test", "Привет").text, "DONE:Привет")
            self.assertEqual(manager.run_agent("agent_1", "работай").text, "DONE:работай")
            self.assertEqual(manager.run_agent("agent_1", "еще").text, "DONE:еще")
            self.assertEqual(manager.agent_status()[0]["state"], "FREE")
            manager.close()
            self.assertTrue(engine.closed)

    def test_builtin_mqtt_tools_are_registered_for_all_agents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.litertlm"
            model.touch()
            manager_prompt = root / "manager.md"
            manager_prompt.write_text("Менеджер", encoding="utf-8")
            (root / "query_manager.md").write_text("QUERY", encoding="utf-8")
            for name in ("agent_1.md", "agent_2.md"):
                (root / name).write_text("Исполнитель", encoding="utf-8")
            config = root / "agents.json"
            config.write_text(json.dumps({"agents": [
                {"id": "agent_1", "prompt": "agent_1.md", "tools": ["shell"]},
                {"id": "agent_2", "prompt": "agent_2.md", "tools": []},
            ]}), encoding="utf-8")
            # Extend an isolated copy of the real JSON without touching
            # production configuration; registration must accept new MQTT tools.
            mqtt_dir = root / "mqtt"
            mqtt_dir.mkdir()
            source = Path(__file__).resolve().parents[1] / "config"
            (mqtt_dir / "mqtt.json").write_text(
                (source / "mqtt.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            specs = json.loads((source / "mqtt_tools.json").read_text(encoding="utf-8"))
            specs["tools"].append({
                "name": "test_added_mqtt_sensor",
                "description": "Read test MQTT occupancy",
                "server": "old_server",
                "topic": "zigbee2mqtt/test_added_mqtt_sensor",
                "fields": {"occupancy": "motion"},
                "validation": {"occupancy": {"type": "boolean"}},
            })
            (mqtt_dir / "mqtt_tools.json").write_text(
                json.dumps(specs), encoding="utf-8",
            )
            mqtt = tuple(entry["name"] for entry in specs["tools"])
            fake = ModuleType("litert_lm")
            fake.Engine = FakeEngine
            fake.Backend = SimpleNamespace(CPU=lambda **kwargs: "cpu")
            fake.SamplerConfig = lambda **kwargs: "sampler"
            fake.ThinkingConfig = lambda **kwargs: "thinking"
            FakeEngine.instances.clear()
            with patch.dict(sys.modules, {"litert_lm": fake}), patch(
                "orchestration_tools.litert_runtime.create_mqtt_tools",
                side_effect=lambda: create_mqtt_tools(mqtt_dir),
            ):
                manager = build_native_manager(
                    model, manager_prompt, agent_config=config,
                    enable_mqtt=True, enable_shell=True, shell_cwd=root,
                )
            try:
                conversations = FakeEngine.instances[0].created
                self.assertEqual(len(conversations), 3)
                names = [
                    [tool.__name__ for tool in item["tools"]]
                    for item in conversations
                ]
                self.assertEqual(manager.shared_tool_names, mqtt)
                for registration in conversations:
                    self.assertIn(
                        "Изображения PNG/JPEG передаются непосредственно",
                        registration["messages"][0]["content"],
                    )
                self.assertEqual(tuple(names[0]), mqtt + ("shell", "delegate_task"))
                self.assertEqual(tuple(names[1]), mqtt + ("shell",))
                self.assertEqual(tuple(names[2]), mqtt)
                added = next(tool for tool in conversations[2]["tools"]
                             if tool.__name__ == "test_added_mqtt_sensor")
                with patch(
                    "litert_agent_tools.mqtt_tools._read_measurement",
                    return_value={"ok": True, "motion": True},
                ) as read:
                    self.assertEqual(added(), {"ok": True, "motion": True})
                self.assertEqual(read.call_args.args[1],
                                 "zigbee2mqtt/test_added_mqtt_sensor")
                # The AGENT's actual registered callable can read real MQTT
                # through the same implementation as the MANAGER.
                agent_weather = next(
                    tool for tool in conversations[1]["tools"]
                    if tool.__name__ == "get_outdoor_weather"
                )
                with patch(
                    "litert_agent_tools.mqtt_tools._read_measurement",
                    return_value={"ok": True, "temperature_c": 8.2,
                                  "humidity_percent": 81.5},
                ) as read:
                    self.assertEqual(agent_weather(), {
                        "ok": True, "temperature_c": 8.2,
                        "humidity_percent": 81.5,
                    })
                self.assertEqual(read.call_count, 1)
                self.assertEqual(read.call_args.args[1], "zigbee2mqtt/temp_ulica")

                # Native vision must remain a direct multimodal message even
                # with a dynamically extended MQTT tool registry.
                image = root / "data" / "vision_probe.png"
                image.parent.mkdir()
                image.write_bytes(bytes.fromhex("89504e470d0a1a0a") + b"fixture")
                with patch.object(
                    manager.chat._working, "send_message",
                    return_value={"role": "assistant", "content": "IMAGE_OK"},
                ) as send:
                    with self.assertLogs(
                        "litert_agent_tools.conversation", level="INFO",
                    ) as logs:
                        self.assertEqual(
                            manager.chat.ask("Что на vision_probe.png?"), "IMAGE_OK",
                        )
                message = send.call_args.args[0]
                self.assertEqual(message["role"], "user")
                self.assertEqual(message["content"][0]["text"], "Что на vision_probe.png?")
                self.assertEqual(
                    message["content"][1],
                    {"type": "image", "path": str(image)},
                )
                self.assertIn("native-vision IMAGE-ATTACHED", "\n".join(logs.output))
            finally:
                manager.close()

    def test_disabled_mqtt_does_not_leak_into_agent_tools(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.litertlm"
            model.touch()
            (root / "manager.md").write_text("Менеджер", encoding="utf-8")
            (root / "agent.md").write_text("Агент", encoding="utf-8")
            (root / "query_manager.md").write_text("QUERY", encoding="utf-8")
            config = root / "agents.json"
            config.write_text(json.dumps({"agents": [
                {"id": "agent_1", "prompt": "agent.md", "tools": ["shell"]},
            ]}), encoding="utf-8")
            fake = ModuleType("litert_lm")
            fake.Engine = FakeEngine
            fake.Backend = SimpleNamespace(CPU=lambda **kwargs: "cpu")
            fake.SamplerConfig = lambda **kwargs: "sampler"
            fake.ThinkingConfig = lambda **kwargs: "thinking"
            FakeEngine.instances.clear()
            with patch.dict(sys.modules, {"litert_lm": fake}):
                manager = build_native_manager(
                    model, root / "manager.md", agent_config=config,
                    enable_mqtt=False, enable_shell=True, shell_cwd=root,
                    vision_backend_name="none",
                )
            try:
                conversations = FakeEngine.instances[0].created
                self.assertEqual(
                    [tool.__name__ for tool in conversations[1]["tools"]],
                    ["shell"],
                )
                self.assertNotIn(
                    "Изображения PNG/JPEG передаются непосредственно",
                    conversations[0]["messages"][0]["content"],
                )
                self.assertNotIn(
                    "Изображения PNG/JPEG передаются непосредственно",
                    conversations[1]["messages"][0]["content"],
                )
            finally:
                manager.close()

    def test_manager_periodic_tools_registered_and_blocked_outside_human_turn(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model.litertlm"
            model.touch()
            (root / "manager.md").write_text("Я MANAGER", encoding="utf-8")
            (root / "query_manager.md").write_text("Я QUERY", encoding="utf-8")
            (root / "agent_1.md").write_text("Я AGENT", encoding="utf-8")
            config = root / "agents.json"
            config.write_text(json.dumps({
                "agents": [{"id": "agent_1", "prompt": "agent_1.md", "tools": []}],
            }), encoding="utf-8")
            fake = ModuleType("litert_lm")
            fake.Engine = FakeEngine
            fake.Backend = SimpleNamespace(CPU=lambda **kwargs: "cpu")
            fake.SamplerConfig = lambda **kwargs: "sampler"
            fake.ThinkingConfig = lambda **kwargs: "thinking"
            FakeEngine.instances.clear()
            with patch.dict(sys.modules, {"litert_lm": fake}), patch(
                "task_system.client.TaskRPC",
            ) as rpc_class:
                rpc = rpc_class.return_value
                rpc.call.return_value = {"task_id": 3}
                manager = build_native_manager(
                    model, root / "manager.md",
                    agent_config=config, task_socket=root / "task.sock",
                    enable_mqtt=True, enable_shell=False,
                )
            try:
                conversations = FakeEngine.instances[0].created
                self.assertEqual(len(conversations), 2)
                names = [
                    [tool.__name__ for tool in item["tools"]]
                    for item in conversations
                ]
                self.assertIn("delegate_task", names[0])
                self.assertNotIn("create_periodic_query", names[0])
                self.assertIn("start_periodic_query", names[0])
                self.assertIn("stop_periodic_query", names[0])
                self.assertIn("delete_periodic_query", names[0])
                self.assertIn("list_periodic_queries", names[0])
                self.assertNotIn("delegate_task", names[1])
                self.assertIn("delegate_task(agent_id, task, period_seconds)", conversations[0]["messages"][0]["content"])
                self.assertIn("start_periodic_query", conversations[0]["messages"][0]["content"])
                self.assertIn("stop_periodic_query", conversations[0]["messages"][0]["content"])
                self.assertIn("delete_periodic_query", conversations[0]["messages"][0]["content"])
                tool = next(tool for tool in conversations[0]["tools"]
                            if tool.__name__ == "delegate_task")
                # Even though registered during BASE warm-up, it cannot
                # mutate TASK SYSTEM until MANAGER enters a human turn.
                self.assertFalse(tool("agent_1", "Измерь температуру", 60)["ok"])
                self.assertFalse(rpc.call.called)

                with patch.object(manager.chat, "ask", side_effect=lambda text: str(
                    tool("agent_1", "Измерь температуру", 60),
                )):
                    result = manager.respond("session", "раз в минуту измеряй температуру")
                self.assertIn("'status': 'scheduled'", result.text)
                rpc.call.assert_called_once()
                self.assertEqual(rpc.call.call_args.args[:2],
                                 ("system", "create_periodic_task"))
                self.assertEqual(rpc.call.call_args.kwargs,
                                 {"method": "query", "executor": "litert", "agent_id": "agent_1"})
                self.assertEqual(rpc.call.call_args.args[3], "Измерь температуру")
                self.assertFalse(tool("agent_1", "Ещё раз", 60)["ok"])
                self.assertEqual(rpc.call.call_count, 1)
            finally:
                manager.close()

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
