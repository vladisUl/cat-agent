from __future__ import annotations

import struct
import unittest

from litert_agent.voice_tts import (
    WAKE_BEEP_SAMPLE_RATE,
    WAKE_BEEP_SECONDS,
    _wake_beep_pcm,
)


def _zero_crossings(pcm: bytes) -> int:
    sample_count = len(pcm) // 2
    samples = struct.unpack(f"<{sample_count}h", pcm)
    return sum(
        (left < 0 <= right) or (left >= 0 > right)
        for left, right in zip(samples, samples[1:])
    )


class WakeBeepTest(unittest.TestCase):
    def test_pcm_has_expected_duration_and_audio(self) -> None:
        pcm = _wake_beep_pcm()
        expected_frames = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_SECONDS)

        self.assertEqual(len(pcm), expected_frames * 2)
        self.assertNotEqual(pcm, b"\x00" * len(pcm))

    def test_tones_start_without_fade(self) -> None:
        pcm = _wake_beep_pcm()
        midpoint = len(pcm) // 2

        first_second_sample = struct.unpack_from("<h", pcm, 2)[0]
        second_second_sample = struct.unpack_from("<h", pcm, midpoint + 2)[0]

        self.assertGreater(abs(first_second_sample), 100)
        self.assertGreater(abs(second_second_sample), 100)

    def test_second_tone_is_clearly_lower_than_first(self) -> None:
        pcm = _wake_beep_pcm()
        midpoint = len(pcm) // 2

        high_crossings = _zero_crossings(pcm[:midpoint])
        low_crossings = _zero_crossings(pcm[midpoint:])

        self.assertGreater(high_crossings, low_crossings)
        self.assertGreater(high_crossings, low_crossings * 1.3)


if __name__ == "__main__":
    unittest.main()
