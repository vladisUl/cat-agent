"""Python-only tests for the opt-in native message JSON tracing proxy."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from litert_agent_tools.native_json_trace import (
    NativeJsonTraceLib,
    attach_native_json_trace,
)


class MockNativeLib:
    def __init__(self, *, fail_on=0):
        self.calls = []
        self.fail_on = fail_on
        self.litert_lm_conversation_delete = lambda ptr: ptr
        self.responses = {}

    def litert_lm_conversation_send_message(self, ptr, raw, ctx, extra):
        self.calls.append((ptr, raw, ctx, extra))
        if self.fail_on == len(self.calls):
            return None
        return 2345

    def litert_lm_json_response_get_string(self, ptr):
        return self.responses[ptr]


class JsonTraceTests(unittest.TestCase):
    def test_proxy_records_user_and_tool_without_mutating_wire(self):
        real = MockNativeLib()
        conversation = SimpleNamespace(_lib=real)
        attach_native_json_trace(conversation)
        self.assertIsInstance(conversation._lib, NativeJsonTraceLib)
        proxy = conversation._lib
        attach_native_json_trace(conversation)
        self.assertIs(proxy, conversation._lib)
        self.assertIs(proxy.litert_lm_conversation_delete, real.litert_lm_conversation_delete)

        user = json.dumps({"role": "user", "content": "выполни ls"})
        tool = json.dumps([{"role": "tool", "content": [{"type": "tool_response", "name": "shell", "response": {"ok": True}}]}])
        with self.assertLogs("litert_agent_tools.native_json_trace", "INFO") as logs:
            self.assertEqual(proxy.litert_lm_conversation_send_message(12, user, "{}", None), 2345)
            self.assertEqual(proxy.litert_lm_conversation_send_message(12, tool, "{}", None), 2345)

        self.assertEqual(real.calls, [(12, user, "{}", None), (12, tool, "{}", None)])
        self.assertIn("role=user valid_python_json=True", logs.output[0])
        self.assertIn("role=['tool'] valid_python_json=True", logs.output[1])

    def test_proxy_logs_exact_message_on_native_failure(self):
        real = MockNativeLib(fail_on=1)
        proxy = NativeJsonTraceLib(real)
        message = json.dumps({"role": "user", "content": "shell ls"})
        with self.assertLogs("litert_agent_tools.native_json_trace", "INFO") as logs:
            self.assertIsNone(proxy.litert_lm_conversation_send_message(12, message, "{}", None))
        self.assertIn("native-json FAILED seq=1", logs.output[-1])
        self.assertIn("valid_python_json=True", logs.output[-1])
        self.assertIn("shell ls", logs.output[-1])

    def test_native_responses_are_preserved_and_empty_final_is_logged(self):
        real = MockNativeLib()
        user_response = b'{"role":"assistant","tool_calls":[{"function":{"name":"shell","arguments":{"command":"ls"}}}]}'
        empty_response = b'{"role":"assistant","content":[],"channels":{"analysis":"done"}}'
        real.responses = {1: user_response, 2: empty_response}
        proxy = NativeJsonTraceLib(real)
        with self.assertLogs("litert_agent_tools.native_json_trace", "INFO") as logs:
            self.assertEqual(proxy.litert_lm_json_response_get_string(1), user_response)
            self.assertEqual(proxy.litert_lm_json_response_get_string(2), empty_response)
        all_logs = "\\n".join(logs.output)
        self.assertIn("RESPONSE seq=1 role=assistant empty_content=True", all_logs)
        self.assertIn("tool_calls=1", all_logs)
        self.assertIn("EMPTY-FINAL seq=2", all_logs)
        self.assertIn("analysis", all_logs)

    def test_proxy_flags_invalid_python_json(self):
        real = MockNativeLib(fail_on=1)
        proxy = NativeJsonTraceLib(real)
        with self.assertLogs("litert_agent_tools.native_json_trace", "INFO") as logs:
            proxy.litert_lm_conversation_send_message(12, "{bad json", "{}", None)
        self.assertIn("valid_python_json=False", logs.output[0])
        self.assertIn("diagnostic_error=", logs.output[1])


if __name__ == "__main__":
    unittest.main()
