from __future__ import annotations

import struct
import unittest

from litert_agent.voice_tts import (
    WAKE_BEEP_SAMPLE_RATE,
    WAKE_BEEP_SECONDS,
    _wake_beep_pcm,
)


class WakeBeepTest(unittest.TestCase):
    def test_pcm_has_expected_duration_and_faded_edges(self) -> None:
        pcm = _wake_beep_pcm()
        expected_frames = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_SECONDS)

        self.assertEqual(len(pcm), expected_frames * 2)
        self.assertEqual(struct.unpack_from("<h", pcm, 0)[0], 0)
        self.assertEqual(struct.unpack_from("<h", pcm, len(pcm) - 2)[0], 0)
        self.assertNotEqual(pcm, b"\x00" * len(pcm))


if __name__ == "__main__":
    unittest.main()
