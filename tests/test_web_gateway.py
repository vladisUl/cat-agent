from __future__ import annotations

import json
import unittest

from litert_agent.web_gateway import _browser_to_core, _encode


class WebGatewayTest(unittest.TestCase):
    def test_user_message_is_normalized_for_core(self) -> None:
        payload = _browser_to_core(json.dumps({"type": "user", "text": "  привет  "}))
        self.assertEqual(payload, {"type": "user", "text": "привет"})

    def test_acquire_client_name_is_forced_to_web(self) -> None:
        payload = _browser_to_core(
            json.dumps({"type": "acquire", "client": "pretend-voice"})
        )
        self.assertEqual(payload, {"type": "acquire", "client": "web"})

    def test_unknown_browser_message_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _browser_to_core(json.dumps({"type": "voice", "text": "обход"}))

    def test_empty_user_message_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _browser_to_core(json.dumps({"type": "user", "text": "   "}))

    def test_binary_websocket_message_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _browser_to_core(b"binary")

    def test_core_wire_format_is_json_line(self) -> None:
        encoded = _encode({"type": "ping"})
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(json.loads(encoded.decode("utf-8")), {"type": "ping"})


if __name__ == "__main__":
    unittest.main()
