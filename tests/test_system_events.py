from __future__ import annotations

import unittest

from cat_agent.system_events import SystemRuntime


class SystemEventsTest(unittest.TestCase):
    def test_periodic_timer_emits_event(self) -> None:
        runtime = SystemRuntime()
        result = runtime.execute("TIMER SET cats 60", "Проверить файл.")
        self.assertTrue(result.startswith("SYSTEM_OK"))

        timer = runtime.timer_snapshot()[0]
        assert timer.next_fire_monotonic is not None
        events = runtime.poll_due(timer.next_fire_monotonic)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].source, "timer")
        self.assertEqual(events[0].name, "cats")
        self.assertEqual(events[0].task, "Проверить файл.")

        timer_after = runtime.timer_snapshot()[0]
        self.assertEqual(timer_after.fired, 1)
        self.assertEqual(timer_after.skipped, 0)

    def test_due_tick_is_skipped_while_runtime_busy(self) -> None:
        runtime = SystemRuntime()
        runtime.execute("TIMER SET cats 60", "Проверить файл.")
        timer = runtime.timer_snapshot()[0]
        assert timer.next_fire_monotonic is not None

        events = runtime.poll_due(timer.next_fire_monotonic, busy=True)
        self.assertEqual(events, ())

        after = runtime.timer_snapshot()[0]
        self.assertEqual(after.fired, 0)
        self.assertEqual(after.skipped, 1)
        self.assertTrue(after.enabled)
        assert after.next_fire_monotonic is not None
        self.assertGreater(after.next_fire_monotonic, timer.next_fire_monotonic)

    def test_timer_control_and_reserved_sources(self) -> None:
        runtime = SystemRuntime()
        runtime.execute("TIMER SET cats 60", "Проверить файл.")
        self.assertTrue(runtime.timer_enabled("cats"))
        self.assertTrue(runtime.execute("TIMER STOP cats", "").startswith("SYSTEM_OK"))
        self.assertFalse(runtime.timer_enabled("cats"))
        self.assertTrue(runtime.execute("TIMER PERIOD cats 30", "").startswith("SYSTEM_OK"))
        self.assertTrue(runtime.execute("TIMER START cats", "").startswith("SYSTEM_OK"))
        self.assertTrue(runtime.execute("TIMER DELETE cats", "").startswith("SYSTEM_OK"))
        self.assertIn("not implemented", runtime.execute("GPIO WATCH 12", ""))


if __name__ == "__main__":
    unittest.main()
