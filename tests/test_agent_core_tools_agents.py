"""Stage 3A: independent AGENT BASEs, explicit tool grants and IPC diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from agent_core_tools.server import CoreServer
from orchestration_tools.agent_config import effective_agent_tool_names, load_agent_specs
from litert_agent_tools.mqtt_tools import BUILTIN_MQTT_TOOL_NAMES, create_mqtt_tools
from orchestration_tools.agents import AgentPool, AgentState, NativeAgent


class FakeChat:
    base_kv_tokens = 250

    def __init__(self):
        self.task_calls: list[str] = []
        self.tokens = 250
        self.reset_count = 0
        self.closed = False
        self.fail = False
        self.empty = False

    @property
    def working_kv_tokens(self):
        return self.tokens

    def ask(self, task):
        self.task_calls.append(task)
        self.tokens += 20
        if self.fail:
            raise RuntimeError("native decode failed")
        return "" if self.empty else f"AGENT: {task}"

    def new_dialogue(self):
        self.reset_count += 1
        self.tokens = self.base_kv_tokens

    def close(self):
        self.closed = True


class AgentPoolTests(unittest.TestCase):
    def test_agent_task_resets_working_context_after_success_and_error(self):
        chat = FakeChat()
        agent = NativeAgent("agent_1", chat)
        self.assertEqual(agent.state, AgentState.FREE)
        first = agent.run("прочти файл")
        self.assertEqual(first.text, "AGENT: прочти файл")
        self.assertEqual(first.kv_before_reset, 270)
        self.assertEqual(agent.state, AgentState.FREE)
        self.assertEqual(chat.working_kv_tokens, 250)
        self.assertEqual(chat.reset_count, 1)
        chat.fail = True
        outcome = agent.run("невыполнимое")
        self.assertEqual(outcome.state, AgentState.FAILED)
        self.assertEqual(agent.state, AgentState.FREE)
        self.assertEqual(chat.reset_count, 2)
        chat.fail = False
        chat.empty = True
        self.assertEqual(agent.run("пустой").state, AgentState.FAILED)
        self.assertEqual(chat.reset_count, 3)
        pool = AgentPool([agent])
        self.assertEqual(pool.status()[0]["base_kv_tokens"], 250)
        pool.close()
        self.assertTrue(chat.closed)

    def test_cooperative_slices_retain_and_then_reset_context(self):
        chat = FakeChat()
        agent = NativeAgent("agent_1", chat)
        pool = AgentPool([agent])
        first = pool.run_slice("agent_1", "first step", final=False)
        self.assertEqual(first.state, AgentState.WAITING)
        self.assertEqual(agent.state, AgentState.WAITING)
        self.assertEqual(chat.working_kv_tokens, 270)
        self.assertEqual(chat.reset_count, 0)
        with self.assertRaisesRegex(RuntimeError, "not available"):
            pool.run("new unrelated task", "agent_1")
        second = pool.run_slice("agent_1", "second step", final=True)
        self.assertEqual(second.state, AgentState.FREE)
        self.assertEqual(chat.task_calls, ["first step", "second step"])
        self.assertEqual(chat.reset_count, 1)
        self.assertEqual(chat.working_kv_tokens, 250)
        self.assertEqual(agent.state, AgentState.FREE)
        pool.close()

    def test_cancel_suspended_agent_without_next_model_call(self):
        chat = FakeChat()
        agent = NativeAgent("agent_1", chat)
        pool = AgentPool([agent])
        pool.run_slice("agent_1", "one", final=False)
        pool.abort_slices("agent_1")
        self.assertEqual(agent.state, AgentState.FREE)
        self.assertEqual(chat.reset_count, 1)
        self.assertEqual(chat.task_calls, ["one"])
        pool.close()

    def test_agent_pool_rejects_unknown_busy_and_duplicate(self):
        one = NativeAgent("agent_1", FakeChat())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            AgentPool([one, one])
        pool = AgentPool([one])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            pool.run("task", "agent_2")
        one.state = AgentState.RUNNING
        with self.assertRaisesRegex(RuntimeError, "not available"):
            pool.run("task", "agent_1")
        one.state = AgentState.FREE
        with self.assertRaises(ValueError):
            pool.run(" ", "agent_1")
        pool.close()

    def test_config_strict_and_agent_specific_tools(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            prompt = root / "agent.md"
            prompt.write_text("Agent prompt", encoding="utf-8")
            config = root / "agents.json"
            config.write_text(json.dumps({"agents": [
                {"id": "agent_1", "prompt": "agent.md", "tools": ["shell"]},
                {"id": "agent_2", "prompt": "agent.md", "tools": []},
            ]}), encoding="utf-8")
            specs = load_agent_specs(config)
            self.assertEqual(specs[0].prompt_file, prompt.resolve())
            self.assertEqual(specs[0].tool_names, ("shell",))
            self.assertEqual(specs[1].tool_names, ())
            for invalid in [
                {"agents": []},
                {"agents": [{"id":"agent_1","prompt":"agent.md","tools":["shell","shell"]}]},
                {"agents": [{"id":"agent_1","prompt":"agent.md","tools": ["shell"]},
                             {"id":"agent_1","prompt":"agent.md","tools": []}]},
                {"agents": [{"id":"agent_1","prompt":"missing.md","tools":[]}]},
            ]:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    config.write_text(json.dumps(invalid), encoding="utf-8")
                    load_agent_specs(config)


    def test_shared_tool_names_are_consistent_for_agent_and_task_worker(self):
        from orchestration_tools.agent_config import AgentSpec

        # The built-in list must not drift away from actual tool callables.
        self.assertEqual(
            BUILTIN_MQTT_TOOL_NAMES,
            tuple(tool.__name__ for tool in create_mqtt_tools()),
        )
        agent = AgentSpec("agent_1", Path("/nonexistent/agent_1.md"),
                          ("shell", "get_outdoor_weather"))
        names = effective_agent_tool_names(agent, BUILTIN_MQTT_TOOL_NAMES)
        self.assertEqual(names, BUILTIN_MQTT_TOOL_NAMES + ("shell",))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(effective_agent_tool_names(agent), agent.tool_names)
        limited = AgentSpec("agent_2", Path("/nonexistent/agent_2.md"), ())
        self.assertEqual(effective_agent_tool_names(limited, BUILTIN_MQTT_TOOL_NAMES),
                         BUILTIN_MQTT_TOOL_NAMES)
        self.assertNotIn("shell",
                         effective_agent_tool_names(limited, BUILTIN_MQTT_TOOL_NAMES))

    def test_external_prompt_directory_overrides_agent_config_location(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            yaml_prompt_root = root / "outside-git" / "prompts_tools"
            yaml_prompt_root.mkdir(parents=True)
            (yaml_prompt_root / "agent_1.md").write_text(
                "EXTERNAL AGENT PROMPT", encoding="utf-8",
            )
            config_dir = root / "cat-agent" / "config"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "native_agents.json"
            config_path.write_text(json.dumps({"agents": [{
                "id": "agent_1", "prompt": "agent_1.md", "tools": [],
            }]}), encoding="utf-8")

            specs = load_agent_specs(config_path, prompt_dir=yaml_prompt_root)
            self.assertEqual(specs[0].prompt_file,
                             (yaml_prompt_root / "agent_1.md").resolve())
            with self.assertRaisesRegex(ValueError, "prompt not found"):
                load_agent_specs(config_path)



class FakeRuntimeManager:
    model_name = "gemma"
    base_kv_tokens = 250
    def __init__(self):
        self.agent = NativeAgent("agent_1", FakeChat())
        self.pool = AgentPool([self.agent])
        self.events = []
    def respond(self, session_id, text):
        from orchestration_tools.contracts import ManagerTurn
        self.events.append("manager")
        return ManagerTurn("reply", "MANAGER " + text)
    def run_agent(self, agent_id, text):
        self.events.append("agent")
        return self.pool.run(text, agent_id)
    def agent_status(self):
        return self.pool.status()
    def release_session(self, session_id):
        self.events.append("release")
    def close(self):
        self.pool.close()


class AgentIpcTests(unittest.TestCase):
    def test_agent_task_uses_same_serialized_core_with_diagnostic_routing(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = FakeRuntimeManager()
            server = CoreServer(Path(temp) / "core.sock", manager=manager)
            server.start()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with socket.socket(socket.AF_UNIX) as sock:
                    sock.settimeout(3)
                    sock.connect(str(server.path))
                    with sock.makefile("r", encoding="utf-8") as rd:
                        def io(msg):
                            sock.sendall((json.dumps(msg, ensure_ascii=False) + "\n").encode())
                            return json.loads(rd.readline())
                        owner = io({"type": "acquire", "client": "probe"})
                        self.assertEqual(owner["type"], "acquired")
                        status = io({"type": "snapshot"})
                        self.assertEqual(status["status"]["agents"][0]["agent_id"], "agent_1")
                        error = io({"type": "agent_task", "agent": "missing", "text": "x"})
                        self.assertEqual(error["type"], "error")
                        self.assertEqual(manager.events, [])
                        for kind, payload in (
                            ("agent_task", {"agent": "agent_1", "text": "вычисли 2+2"}),
                            ("user", {"text": "hello"}),
                            ("agent_task", {"agent": "agent_1", "text": "второе задание"}),
                        ):
                            first = io({"type": kind, **payload})
                            reply = json.loads(rd.readline())
                            done = json.loads(rd.readline())
                            self.assertEqual([first["type"],reply["type"],done["type"]],
                                             ["accepted","reply","completed"])
                            self.assertEqual(reply["request_id"],done["request_id"])
                            if kind == "agent_task":
                                self.assertEqual(reply["agent_id"], "agent_1")
                                self.assertFalse(reply["chat_open"])
                                self.assertEqual(manager.agent.chat.working_kv_tokens, 250)
                            else:
                                self.assertEqual(reply["text"], "MANAGER hello")
                        self.assertEqual(manager.events, ["agent","manager","agent"])
            finally:
                server.close()
                thread.join(timeout=2)
            self.assertTrue(manager.agent.chat.closed)


if __name__ == "__main__":
    unittest.main()
