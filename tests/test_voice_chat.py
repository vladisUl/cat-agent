import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from litert_agent import voice


class VoiceChatTest(unittest.TestCase):
    def test_three_turn_chat_requires_only_one_wake_word(self):
        wake = Mock()
        wake.AcceptWaveform.return_value = True
        wake.Result.return_value = json.dumps({"text": "гена"})
        command = Mock()
        command.AcceptWaveform.return_value = True
        command.Result.return_value = json.dumps({"text": "речь"})
        pcm = Mock()
        pcm.read.side_effect = [(800, b"\0" * 1600)] * 4 + [KeyboardInterrupt()]
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(voice, "COMMAND_WAV_PATH", Path(temp) / "command.wav"), \
             patch.object(voice, "_open_microphone", return_value=pcm), \
             patch.object(voice, "_make_wake_recognizer", return_value=wake), \
             patch.object(voice, "_make_command_recognizer", return_value=command), \
             patch.object(voice, "play_wake_beep") as beep, \
             patch.object(voice, "_transcribe_command", side_effect=["Чат", "температура", "Конец чата"]), \
             patch.object(voice, "_run_core_voice_turn", side_effect=[
                 voice.VoiceTurnResult("Чат создан", True),
                 voice.VoiceTurnResult("20 градусов", True),
                 voice.VoiceTurnResult("Чат закрыт", False),
             ]) as run:
            with self.assertRaises(KeyboardInterrupt):
                voice._run_loop(SimpleNamespace(ALSAAudioError=RuntimeError), None, None, None, None)
            self.assertEqual(wake.AcceptWaveform.call_count, 1)
            self.assertEqual(beep.call_count, 1)
            self.assertEqual([call.args[0] for call in run.call_args_list], ["Чат", "температура", "Конец чата"])

    def test_wake_reopens_capture_before_beep(self):
        events = []
        wake = Mock()
        wake.AcceptWaveform.return_value = True
        wake.Result.return_value = json.dumps({"text": "гена"})
        command = Mock()

        first_pcm = Mock()
        first_pcm.read.return_value = (800, b"\0" * 1600)
        first_pcm.close.side_effect = lambda: events.append("close")

        second_pcm = Mock()
        second_pcm.read.side_effect = KeyboardInterrupt()
        second_pcm.close.side_effect = lambda: events.append("final-close")

        captures = iter([first_pcm, second_pcm])

        def open_microphone(_alsa):
            events.append("open")
            return next(captures)

        def beep():
            events.append("beep")

        with tempfile.TemporaryDirectory() as temp, \
             patch.object(voice, "COMMAND_WAV_PATH", Path(temp) / "command.wav"), \
             patch.object(voice, "_open_microphone", side_effect=open_microphone), \
             patch.object(voice, "_make_wake_recognizer", return_value=wake), \
             patch.object(voice, "_make_command_recognizer", return_value=command), \
             patch.object(voice, "play_wake_beep", side_effect=beep):
            with self.assertRaises(KeyboardInterrupt):
                voice._run_loop(SimpleNamespace(ALSAAudioError=RuntimeError), None, None, None, None)

        self.assertEqual(events[:4], ["open", "close", "open", "beep"])
