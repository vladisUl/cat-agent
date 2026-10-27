"""Isolated tests: no LiteRT native library, running MQTT broker or model needed."""

from __future__ import annotations

import ctypes
from importlib import metadata
import json
import subprocess
import sys
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from litert_agent_tools.clone import ConversationCloneError, clone_conversation
from litert_agent_tools.config import MqttConfig
from litert_agent_tools.conversation import (
    InvalidUserTextError, NativeConversationChat, response_text,
)
from litert_agent_tools.mqtt_tools import TOPICS, create_mqtt_tools
from litert_agent_tools.native_patch.enable_prefill import CHANGES, change_source


class MqttToolTests(unittest.TestCase):
    def setUp(self):
        self.config = MqttConfig(host="localhost", port=1884, wait_seconds=3)
        self.tools = create_mqtt_tools(self.config)

    def test_tools_are_zero_arg_and_named_for_model(self):
        self.assertEqual([t.__name__ for t in self.tools], [
            "get_outdoor_weather", "get_home_climate",
            "get_aquarium_temperature", "get_freezer_temperature",
        ])
        self.assertEqual(len(TOPICS), 4)
        self.assertTrue(all(t.__doc__ for t in self.tools))

    def test_reads_exact_topic_and_normalizes_payload(self):
        payloads = [
            {"temperature": 16.75, "humidity": 60.2},
            {"temperature": 23, "humidity": 41},
            {"temperature": 27.1},
            {"temperature": -18.5},
        ]
        with patch(
            "litert_agent_tools.mqtt_tools.subprocess.run",
            side_effect=[
                subprocess.CompletedProcess([], 0, json.dumps(p), "")
                for p in payloads
            ],
        ) as run:
            results = [tool() for tool in self.tools]
        self.assertTrue(all(r["ok"] for r in results))
        self.assertEqual(results[0]["humidity_percent"], 60.2)
        self.assertEqual(results[2]["temperature_c"], 27.1)
        self.assertEqual(results[3]["temperature_c"], -18.5)
        for index, call in enumerate(run.call_args_list):
            cmd = call.args[0]
            self.assertEqual(cmd[cmd.index("-t") + 1], list(TOPICS.values())[index][0])
            self.assertEqual(cmd[cmd.index("-h") + 1], "localhost")
            self.assertEqual(cmd[cmd.index("-W") + 1], "3")
            self.assertNotIn("shell", call.kwargs)

    def test_empty_bad_missing_and_invalid_measurements(self):
        for payload in ("", "[]", "{}", '{"temperature":true,"humidity":60}',
                        '{"temperature":20,"humidity":105}', "not json"):
            with self.subTest(payload=payload), patch(
                "litert_agent_tools.mqtt_tools.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, payload, ""),
            ):
                self.assertFalse(self.tools[0]()["ok"])

    def test_logs_real_tool_call_and_mqtt_result(self):
        with patch(
            "litert_agent_tools.mqtt_tools.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, '{"temperature":24.66,"humidity":26.24}', ""),
        ), self.assertLogs("litert_agent_tools.mqtt_tools", level="INFO") as captured:
            result = self.tools[1]()
        self.assertTrue(result["ok"])
        self.assertIn("native-tool CALL name=get_home_climate topic=zigbee2mqtt/temp_tua", captured.output[0])
        self.assertIn("native-tool RESULT name=get_home_climate", captured.output[1])
        self.assertIn("24.66", captured.output[1])
        self.assertIn("26.24", captured.output[1])

    def test_missing_mqtt_binary_and_timeout(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired(["mosquitto_sub"], 3)):
            with patch("litert_agent_tools.mqtt_tools.subprocess.run", side_effect=error):
                self.assertFalse(self.tools[0]()["ok"])

    def test_nonzero_exit_is_reported(self):
        with patch(
            "litert_agent_tools.mqtt_tools.subprocess.run",
            return_value=subprocess.CompletedProcess([], 27, "", "Timed out"),
        ):
            result = self.tools[0]()
        self.assertFalse(result["ok"])
        self.assertIn("27", result["error"])

    def test_config_validation(self):
        for values in ({"port": 0}, {"wait_seconds": 0}, {"host": " "}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MqttConfig(**values)


class StubConversation:
    def __init__(self, messages, tools):
        self.messages = list(messages)
        self.tools = tools
        self.closed = False
        self.calls = []
        self.token_count = 0

    def clone(self):
        clone = StubConversation(self.messages, self.tools)
        clone.calls = list(self.calls)
        clone.token_count = self.token_count
        return clone

    def send_message(self, question):
        self.calls.append(question)
        self.token_count += len(question) + 8
        self.messages.append({"role": "user", "content": question})
        text = f"answer: {question}"
        self.messages.append({"role": "assistant", "content": text})
        return {"role": "assistant", "content": [{"type": "text", "text": text}]}

    def close(self):
        self.closed = True


class StubEngine:
    def __init__(self):
        self.base = None
        self.options = None

    def create_conversation(self, **kwargs):
        self.options = kwargs
        self.base = StubConversation(kwargs["messages"], kwargs["tools"])
        return self.base


class WarmStubConversation(StubConversation):
    def __init__(self, messages, tools, *, tokens=275):
        super().__init__(messages, tools)
        self.token_count = tokens

    def clone(self):
        result = WarmStubConversation(self.messages, self.tools, tokens=self.token_count)
        result.calls = list(self.calls)
        return result

    def get_benchmark_info(self):
        return SimpleNamespace(last_prefill_token_count=7)


class WarmStubEngine:
    def __init__(self, tokens=275):
        self.tokens = tokens
        self.base = None
        self.prefill_requested = False

    def create_conversation(self, *, messages, tools, automatic_tool_calling,
                            prefill_preface_on_init=False, **kwargs):
        self.prefill_requested = prefill_preface_on_init
        self.base = WarmStubConversation(messages, tools, tokens=self.tokens)
        return self.base


class ConversationTests(unittest.TestCase):
    def test_base_clone_reset_history_and_cleanup(self):
        engine = StubEngine()
        with NativeConversationChat(engine, "TEMP PROMPT", ["tool"]) as chat:
            self.assertTrue(engine.options["automatic_tool_calling"])
            self.assertEqual(engine.base.messages[0]["content"], "TEMP PROMPT")
            self.assertTrue(chat.base_prefilled)
            self.assertEqual(engine.base.calls, ["warm"])
            self.assertEqual(chat.base_warm_reply, "answer: warm")
            self.assertGreater(chat.base_kv_tokens, 0)
            self.assertEqual(chat._working.token_count, chat.base_kv_tokens)
            self.assertEqual(chat.ask("first"), "answer: first")
            first = chat._working
            self.assertEqual(chat.ask("second"), "answer: second")
            chat.new_dialogue()
            self.assertTrue(first.closed)
            self.assertEqual(chat._working.calls, ["warm"])
            self.assertEqual(chat._working.token_count, chat.base_kv_tokens)
            self.assertEqual(chat.ask("third"), "answer: third")
            self.assertEqual(engine.base.calls, ["warm"])
        self.assertTrue(engine.base.closed)

    def test_stock_litert_warm_turn_and_kv_inheritance(self):
        engine = WarmStubEngine(tokens=0)
        with self.assertLogs("litert_agent_tools.conversation", level="INFO") as events:
            with NativeConversationChat(engine, "TEMP PROMPT", ["tool"], warm_message="warm") as chat:
                self.assertFalse(engine.prefill_requested)
                self.assertTrue(chat.base_prefilled)
                self.assertEqual(engine.base.calls, ["warm"])
                self.assertEqual(chat.base_warm_reply, "answer: warm")
                warm_tokens = chat.base_kv_tokens
                self.assertEqual(warm_tokens, 12)
                self.assertEqual(chat._working.token_count, warm_tokens)
                self.assertEqual(chat.ask("test"), "answer: test")
                chat.new_dialogue()
                self.assertEqual(chat._working.token_count, warm_tokens)
                self.assertEqual(engine.base.calls, ["warm"])
        logs = "\n".join(events.output)
        self.assertIn("BASE-WARM message='warm' response='answer: warm'", logs)
        self.assertIn("kv_before=0 kv_after=12", logs)
        self.assertIn("BASE-CLONE kv_tokens=12 expected=12", logs)
        self.assertIn("last_prefill_tokens=7", logs)

    def test_warm_rejects_missing_native_kv(self):
        engine = WarmStubEngine(tokens=0)
        with patch.object(WarmStubConversation, "send_message",
                          return_value={"role": "assistant", "content": "OK"}):
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                NativeConversationChat(engine, "TEMP PROMPT", [])
        self.assertTrue(engine.base.closed)

    def test_warm_rejects_kv_that_does_not_increase(self):
        engine = WarmStubEngine(tokens=275)
        with patch.object(WarmStubConversation, "send_message",
                          return_value={"role": "assistant", "content": "OK"}):
            with self.assertRaisesRegex(RuntimeError, "did not increase"):
                NativeConversationChat(engine, "TEMP PROMPT", [])
        self.assertTrue(engine.base.closed)

    def test_warm_rejects_empty_reply(self):
        engine = WarmStubEngine(tokens=0)
        with patch.object(WarmStubConversation, "send_message",
                          return_value={"role": "assistant", "content": " "}):
            with self.assertRaisesRegex(RuntimeError, "empty assistant reply"):
                NativeConversationChat(engine, "TEMP PROMPT", [])
        self.assertTrue(engine.base.closed)

    def test_warm_exception_does_not_clone_and_closes_base(self):
        engine = WarmStubEngine(tokens=0)
        with patch.object(WarmStubConversation, "send_message",
                          side_effect=RuntimeError("native failure")):
            with patch("litert_agent_tools.conversation.clone_conversation") as cloned:
                with self.assertRaisesRegex(RuntimeError, "native failure"):
                    NativeConversationChat(engine, "TEMP PROMPT", [])
        cloned.assert_not_called()
        self.assertTrue(engine.base.closed)

    def test_warm_rejects_clone_with_missing_kv(self):
        engine = WarmStubEngine(tokens=275)
        with patch("litert_agent_tools.conversation.clone_conversation",
                   return_value=WarmStubConversation([], [], tokens=0)) as cloned:
            with self.assertRaisesRegex(RuntimeError, "does not match warmed BASE"):
                NativeConversationChat(engine, "TEMP PROMPT", [])
        self.assertEqual(cloned.call_count, 1)
        self.assertTrue(engine.base.closed)

    def test_warm_message_can_be_custom_without_an_ok_reply_requirement(self):
        engine = StubEngine()
        with NativeConversationChat(engine, "TEMP PROMPT", [], warm_message="startup") as chat:
            self.assertEqual(engine.base.calls, ["startup"])
            self.assertEqual(chat.base_warm_reply, "answer: startup")
            self.assertTrue(chat.base_prefilled)

    def test_reset_clone_error_does_not_destroy_active_dialogue(self):
        engine = StubEngine()
        with NativeConversationChat(engine, "temporary", []) as chat:
            current = chat._working
            with patch("litert_agent_tools.conversation.clone_conversation", side_effect=RuntimeError("clone failed")):
                with self.assertRaisesRegex(RuntimeError, "clone failed"):
                    chat.new_dialogue()
            self.assertIs(chat._working, current)
            self.assertFalse(current.closed)

    def test_invalid_unicode_is_rejected_before_native_call_without_reset(self):
        engine = StubEngine()
        with NativeConversationChat(engine, "TEMP PROMPT", []) as chat:
            resident = chat._working
            tokens = resident.token_count
            calls = list(resident.calls)
            for question in (
                "выполни " + chr(0xDCD1) + "shell ls",
                "bad" + chr(0xD800),
                "bad" + chr(0xDFFF),
            ):
                with self.subTest(question=repr(question)), self.assertRaisesRegex(
                    InvalidUserTextError, "Invalid Unicode U+"
                ):
                    chat.ask(question)
                self.assertIs(chat._working, resident)
                self.assertFalse(resident.closed)
                self.assertEqual(resident.calls, calls)
                self.assertEqual(resident.token_count, tokens)
            self.assertEqual(chat.ask("выполни shell ls"), "answer: выполни shell ls")

    def test_invalid_input_and_response_shapes(self):
        engine = StubEngine()
        with self.assertRaises(ValueError):
            NativeConversationChat(engine, " ", [])
        with self.assertRaises(ValueError):
            NativeConversationChat(engine, "temp", [], warm_message=" ")
        with NativeConversationChat(engine, "temp", []) as chat:
            with self.assertRaises(ValueError):
                chat.ask(" ")
        self.assertEqual(response_text({"content": "hello"}), "hello")
        self.assertEqual(response_text({"content": ["a", {"type": "text", "text": "b"}]}), "ab")
        self.assertEqual(response_text({"content": [{"type": "text", "text": "A"},
                                                     {"type": "other", "text": "B"}]}), "A")


class CloneAdapterTests(unittest.TestCase):
    def test_uses_new_python_method_when_present(self):
        source = SimpleNamespace(clone=lambda: "native-clone")
        self.assertEqual(clone_conversation(source), "native-clone")

    def test_unknown_legacy_abi_is_rejected(self):
        with patch("litert_agent_tools.clone.metadata.version", return_value="1.0.0"):
            with self.assertRaises(ConversationCloneError):
                clone_conversation(SimpleNamespace())

    def test_legacy_c_pointer_and_python_metadata(self):
        calls = []
        def native_clone(ptr):
            calls.append(ptr)
            return 8675

        native_clone.argtypes = None
        native_clone.restype = None
        fake_lib = SimpleNamespace(litert_lm_conversation_clone=native_clone)
        source = SimpleNamespace(
            _ptr=123, _lib=fake_lib, _engine="engine",
            messages=[{"role": "system", "content": "test"}], tools=["tool"],
            _tools_map={"tool": "object"}, tool_event_handler=None,
            automatic_tool_calling=True, extra_context={},
            thinking_config=None, sampler_config="sampler",
            lora_config=None, max_output_tokens=123,
            chat_template=None, constrained_decoding_config=None,
        )
        class FakeWrapper:
            def __init__(self, lib, ptr, **kwargs):
                self.lib, self.ptr, self.kwargs = lib, ptr, kwargs

        package = ModuleType("litert_lm")
        module = ModuleType("litert_lm.conversation")
        module.Conversation = FakeWrapper
        with patch("litert_agent_tools.clone.metadata.version", return_value="0.18.0"), \
             patch.dict(sys.modules, {"litert_lm": package, "litert_lm.conversation": module}):
            result = clone_conversation(source)

        self.assertEqual(result.ptr, 8675)
        self.assertEqual(calls, [123])
        self.assertEqual(native_clone.argtypes, [ctypes.c_void_p])
        self.assertEqual(native_clone.restype, ctypes.c_void_p)
        self.assertEqual(result.kwargs["tools_map"], {"tool": "object"})
        self.assertTrue(result.kwargs["automatic_tool_calling"])


class NativeSourcePatchTests(unittest.TestCase):
    def test_patch_is_checked_and_idempotent(self):
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            for relative, changes in CHANGES.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(before for before, _ in changes), encoding="utf-8")
            before = (root / "c/conversation.cc").read_text(encoding="utf-8")
            status = change_source(root, check=True)
            self.assertTrue(all(line.startswith("PATCH") for line in status))
            self.assertEqual((root / "c/conversation.cc").read_text(encoding="utf-8"), before)
            change_source(root, check=False)
            self.assertTrue(all(line.startswith("OK") for line in change_source(root, check=False)))
            self.assertIn("SetPrefillPrefaceOnInit", (root / "c/conversation.cc").read_text(encoding="utf-8"))
            self.assertIn("prefill_preface_on_init: bool = False",
                          (root / "python/litert_lm/engine.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
