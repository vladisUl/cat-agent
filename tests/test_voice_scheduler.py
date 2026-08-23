from __future__ import annotations

from types import SimpleNamespace
import unittest

from litert_agent.core_scheduler import MANAGER_PRIORITY
from litert_agent.voice_scheduler import (
    VOICE_PRIORITY,
    VOICE_REQUEST_LABEL,
    VoiceCoreScheduler,
)


class VoiceSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler = VoiceCoreScheduler(SimpleNamespace())

    def tearDown(self) -> None:
        self.scheduler._executor.shutdown(wait=True, cancel_futures=True)

    def test_voice_outranks_normal_user_request(self) -> None:
        self.scheduler.submit_user("обычный запрос")
        self.scheduler.submit_voice("голосовой запрос")

        first = self.scheduler._take_next_request()
        second = self.scheduler._take_next_request()

        assert first is not None
        assert second is not None
        self.assertEqual(first.label, VOICE_REQUEST_LABEL)
        self.assertEqual(first.priority, VOICE_PRIORITY)
        self.assertEqual(first.payload, "голосовой запрос")
        self.assertEqual(second.label, "user")
        self.assertEqual(second.priority, MANAGER_PRIORITY)
        self.assertLess(VOICE_PRIORITY, MANAGER_PRIORITY)


if __name__ == "__main__":
    unittest.main()
