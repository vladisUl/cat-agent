from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from litert_agent.firebase_gateway import _decode, _encode, read_tokens


class FirebaseGatewayTest(unittest.TestCase):
    def test_read_tokens_deduplicates_by_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokens.txt"
            path.write_text(
                "\n".join(
                    (
                        json.dumps({"id": "phone", "token": "old"}),
                        json.dumps({"id": "tablet", "token": "tab"}),
                        json.dumps({"id": "phone", "token": "new"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(read_tokens(path), ["new", "tab"])

    def test_read_tokens_rejects_bad_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tokens.txt"
            path.write_text('{"id":"phone"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_tokens(path)

    def test_core_wire_format_is_json_line(self) -> None:
        payload = {"type": "register_fallback", "client": "firebase"}
        encoded = _encode(payload)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(_decode(encoded.decode("utf-8")), payload)

    def test_decode_rejects_non_object(self) -> None:
        with self.assertRaises(ValueError):
            _decode('["notification"]')


if __name__ == "__main__":
    unittest.main()
