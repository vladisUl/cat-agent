from __future__ import annotations

import unittest
from unittest.mock import patch

from litert_agent.voice_tts import (
    WAKE_BEEP_FREQUENCY_HZ,
    WAKE_BEEP_SAMPLE_RATE,
    WAKE_BEEP_SECONDS,
    _wake_beep_pcm,
    play_wake_beep,
)


class WakeBeepTest(unittest.TestCase):
    def test_pcm_has_expected_duration_and_audio(self) -> None:
        pcm = _wake_beep_pcm()
        expected_frames = round(WAKE_BEEP_SAMPLE_RATE * WAKE_BEEP_SECONDS)

        self.assertEqual(len(pcm), expected_frames * 2)
        self.assertNotEqual(pcm, b"\x00" * len(pcm))
        self.assertEqual(WAKE_BEEP_FREQUENCY_HZ, 784.0)
        self.assertEqual(WAKE_BEEP_SECONDS, 0.50)

    @patch("litert_agent.voice_tts.subprocess.run")
    def test_playback_uses_one_aplay_call(self, run_mock) -> None:
        play_wake_beep()

        run_mock.assert_called_once()
        pcm = run_mock.call_args.kwargs["input"]
        self.assertEqual(pcm, _wake_beep_pcm())


if __name__ == "__main__":
    unittest.main()
