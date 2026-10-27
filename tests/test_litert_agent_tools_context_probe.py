"""Pure-Python tests of the A/B native context probe (no LiteRT Engine required)."""
from __future__ import annotations

import re
import unittest
from unittest.mock import patch

from orchestration_tools.probe_contexts import _words, run_probe


class FakeConversation:
    def __init__(self, *, base=323, other=None):
        self.token_count = base
        self.secret = None
        self.closed = False
        self.response_override = None
        self.other = other
        self.corrupt_other_kv = False

    def send_message(self, question):
        self.token_count += len(question) + 7
        match = re.search(r"секретное слово — ([А-ЯЁ]+)", question)
        if match:
            self.secret = match.group(1)
            answer = "ПРИНЯТО"
        else:
            answer = self.response_override or self.secret or "НЕИЗВЕСТНО"
        if self.corrupt_other_kv and self.other is not None:
            self.other.token_count += 1
        return {"role": "assistant", "content": [{"type": "text", "text": answer}]}

    def close(self):
        self.closed = True


class FakeChat:
    base_kv_tokens = 323

    def __init__(self):
        self.a = FakeConversation()
        self.b = FakeConversation(other=self.a)
        self.a.other = self.b

    @property
    def working_kv_tokens(self):
        return self.a.token_count

    @property
    def base_resident_kv_tokens(self):
        return self.base_kv_tokens

    def ask(self, question):
        return self.a.send_message(question)["content"][0]["text"]

    def fork_dialogue(self):
        return self.b


class ProbeTests(unittest.TestCase):
    def test_semantic_word_normalization(self):
        self.assertEqual(_words("«КЕДР»."), ["кедр"])
        self.assertEqual(_words("Мой ответ — КЕДР"), ["мой", "ответ", "кедр"])

    def test_two_conversations_keep_independent_code_and_kv(self):
        class Manager: pass
        manager = Manager()
        manager.chat = FakeChat()
        with patch("builtins.print"):
            kv_ok, semantic_ok = run_probe(manager, codes=("КЕДР", "САПФИР"))
        self.assertTrue(kv_ok)
        self.assertTrue(semantic_ok)
        self.assertTrue(manager.chat.b.closed)
        self.assertEqual(manager.chat.a.secret, "КЕДР")
        self.assertEqual(manager.chat.b.secret, "САПФИР")

    def test_detect_model_recall_failure_without_blame_on_kv(self):
        class Manager: pass
        manager = Manager()
        manager.chat = FakeChat()
        manager.chat.b.response_override = "НЕИЗВЕСТНО"
        with patch("builtins.print"):
            kv_ok, semantic_ok = run_probe(manager, codes=("КЕДР", "САПФИР"))
        self.assertTrue(kv_ok)
        self.assertFalse(semantic_ok)

    def test_detect_cross_conversation_kv_change(self):
        class Manager: pass
        manager = Manager()
        manager.chat = FakeChat()
        manager.chat.b.corrupt_other_kv = True
        with patch("builtins.print"):
            kv_ok, semantic_ok = run_probe(manager, codes=("КЕДР", "САПФИР"))
        self.assertFalse(kv_ok)
        self.assertTrue(semantic_ok)
        self.assertTrue(manager.chat.b.closed)


if __name__ == "__main__":
    unittest.main()
