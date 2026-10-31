"""Native Conversation image routing without native tool calls or extra prompts."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from litert_agent_tools.conversation import NativeConversationChat
from litert_agent_tools.vision_input import native_picture_message, picture_path_in_text


PNG = bytes.fromhex("89504e470d0a1a0a") + b"file"
JPEG = bytes.fromhex("ffd8ff") + b"file"


class PictureInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "data" / "test.png").write_bytes(PNG)
        (self.root / "data" / "other.jpg").write_bytes(JPEG)

    def test_no_image_keeps_text_unchanged(self):
        question = "Какая сейчас погода?"
        self.assertEqual(native_picture_message(question, self.root), question)

    def test_question_and_native_path_are_preserved(self):
        question = "Посмотри test.png. Сколько на фотографии людей?"
        message = native_picture_message(question, self.root)
        self.assertEqual(message["role"], "user")
        self.assertEqual(message["content"][0], {"type": "text", "text": question})
        self.assertEqual(message["content"][1],
                         {"type": "image", "path": str(self.root / "data" / "test.png")})
        self.assertNotIn("blob", str(message))
        self.assertEqual(len(message["content"]), 2)

    def test_legacy_directive_and_jpeg(self):
        question = '/work#read_pic.sh other.jpg'
        message = native_picture_message(question, self.root)
        self.assertEqual(message["content"][0]["text"], question)
        self.assertEqual(message["content"][1]["path"], str(self.root / "data" / "other.jpg"))
        self.assertEqual(picture_path_in_text('read_pic.sh "my photo.png"'), "my photo.png")

    def test_missing_file_rejected_and_workspace_escape_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            native_picture_message("Посмотри absent.png", self.root)
        with self.assertRaisesRegex(ValueError, "invalid data path"):
            native_picture_message("/work#read_pic.sh ../escape.png", self.root)

    def test_multiple_image_paths_require_explicit_one(self):
        with self.assertRaisesRegex(ValueError, "one PNG/JPEG"):
            native_picture_message("Сравни test.png и other.jpg", self.root)

    def test_uses_legacy_size_limit(self):
        with patch.dict("os.environ", {"CAT_AGENT_MAX_IMAGE_BYTES": "4"}):
            with self.assertRaisesRegex(ValueError, "image exceeds"):
                native_picture_message("Посмотри test.png", self.root)

    def test_send_once_as_original_question_plus_image(self):
        class FakeConversation:
            def __init__(self, calls=None, count=0):
                self.calls = [] if calls is None else calls
                self.token_count = count
                self.closed = False
                self.automatic_tool_calling = True
                self.tool_modes = []

            def send_message(self, payload):
                self.tool_modes.append(self.automatic_tool_calling)
                self.calls.append(payload)
                self.token_count += 10
                return {"content": "OK"}

            def clone(self):
                return FakeConversation(count=self.token_count)

            def close(self):
                self.closed = True

        class FakeEngine:
            def create_conversation(self, **kwargs):
                return FakeConversation()

        with NativeConversationChat(
            FakeEngine(), "system", [], picture_root=self.root,
        ) as chat:
            question = "Что изображено на test.png и какого оно цвета?"
            answer = chat.ask(question)
            self.assertEqual(answer, "OK")
            self.assertEqual(chat.execution_tool_calls, 0)
            self.assertEqual(chat._working.calls[0]["content"][0]["text"], question)
            self.assertEqual(chat._working.calls[0]["content"][1]["type"], "image")
            self.assertEqual(len(chat._working.calls), 1)
            self.assertEqual(chat._working.tool_modes, [False])
            self.assertTrue(chat._working.automatic_tool_calling)


if __name__ == "__main__":
    unittest.main()
