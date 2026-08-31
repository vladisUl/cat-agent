from __future__ import annotations

import struct
import unittest
from unittest.mock import patch

from litert_agent.voice_tts import (
    WAKE_BEEP_PAUSE_SECONDS,
    WAKE_BEEP_SAMPLE_RATE,
    WAKE_BEEP_SECONDS,
    WAKE_BEEP_TONE_SECONDS,
    _wake_beep_pcm,
    play_wake_beep,
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

    def test_tones_are_separated_by_silence(self) -> None:
        pcm = _wake_beep_pcm()
        tone_bytes = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_TONE_SECONDS) * 2
        pause_bytes = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_PAUSE_SECONDS) * 2

        pause = pcm[tone_bytes : tone_bytes + pause_bytes]
        self.assertEqual(pause, b"\x00" * pause_bytes)

        first_second_sample = struct.unpack_from("<h", pcm, 2)[0]
        second_tone_offset = tone_bytes + pause_bytes
        second_second_sample = struct.unpack_from("<h", pcm, second_tone_offset + 2)[0]

        self.assertGreater(abs(first_second_sample), 100)
        self.assertGreater(abs(second_second_sample), 100)

    def test_second_tone_is_clearly_lower_than_first(self) -> None:
        pcm = _wake_beep_pcm()
        tone_bytes = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_TONE_SECONDS) * 2
        pause_bytes = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_PAUSE_SECONDS) * 2

        high_pcm = pcm[:tone_bytes]
        low_pcm = pcm[tone_bytes + pause_bytes :]

        high_crossings = _zero_crossings(high_pcm)
        low_crossings = _zero_crossings(low_pcm)

        self.assertGreater(high_crossings, low_crossings)
        self.assertGreater(high_crossings, low_crossings * 1.3)

    @patch("litert_agent.voice_tts.time.sleep")
    @patch("litert_agent.voice_tts.subprocess.run")
    def test_playback_uses_two_separate_aplay_calls(self, run_mock, sleep_mock) -> None:
        play_wake_beep()

        self.assertEqual(run_mock.call_count, 2)
        sleep_mock.assert_called_once_with(WAKE_BEEP_PAUSE_SECONDS)

        first_pcm = run_mock.call_args_list[0].kwargs["input"]
        second_pcm = run_mock.call_args_list[1].kwargs["input"]

        self.assertGreater(_zero_crossings(first_pcm), _zero_crossings(second_pcm))


if __name__ == "__main__":
    unittest.main()
