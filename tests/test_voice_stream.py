from __future__ import annotations

import unittest

from litert_agent.voice_stream import ManagerVoiceStream, TextFragmenter


class ManagerVoiceStreamTest(unittest.TestCase):
    def test_chunked_reply_prefix_becomes_visible(self) -> None:
        stream = ManagerVoiceStream()
        stream.decode_start()

        self.assertEqual(stream.feed_chunk("RE"), "")
        self.assertEqual(stream.feed_chunk("PLY\nПривет"), "Привет")
        self.assertEqual(stream.feed_chunk(", Влад."), ", Влад.")

    def test_work_command_stays_hidden_until_next_decode(self) -> None:
        stream = ManagerVoiceStream()
        stream.decode_start()

        self.assertEqual(stream.feed_chunk("/work#shell"), "")
        self.assertEqual(stream.feed_chunk("\nls"), "")

        stream.decode_start()
        self.assertEqual(stream.feed_chunk("REPLY\nГотово."), "Готово.")

    def test_ask_is_human_facing(self) -> None:
        stream = ManagerVoiceStream()
        stream.decode_start()

        self.assertEqual(stream.feed_chunk("ASK\nКакой датчик?"), "Какой датчик?")


class TextFragmenterTest(unittest.TestCase):
    def test_first_complete_sentence_is_emitted_immediately(self) -> None:
        fragmenter = TextFragmenter()
        self.assertEqual(fragmenter.feed("Первая фраза. "), ["Первая фраза."])

    def test_tail_is_flushed_on_finish(self) -> None:
        fragmenter = TextFragmenter()
        self.assertEqual(fragmenter.feed("Короткий ответ без точки"), [])
        self.assertEqual(fragmenter.finish(), ["Короткий ответ без точки"])


if __name__ == "__main__":
    unittest.main()
