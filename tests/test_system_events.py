from __future__ import annotations

import unittest

from cat_agent.protocol import ManagerAction, parse_manager_output
from cat_agent.system_events import SystemRuntime


class SystemEventsTest(unittest.TestCase):
    def test_manager_system_directive(self) -> None:
        item = parse_manager_output(
            "SYSTEM TIMER SET cats 60\nПроверить файл с наблюдениями."
        )
        self.assertEqual(item.action, ManagerAction.SYSTEM)
        self.assertEqual(item.system_command, "TIMER SET cats 60")
        self.assertEqual(item.body, "Проверить файл с наблюдениями.")

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
        assert timer_after.next_fire_monotonic is not None
        self.assertGreater(
            timer_after.next_fire_monotonic,
            timer.next_fire_monotonic,
        )

    def test_timer_control_and_reserved_sources(self) -> None:
        runtime = SystemRuntime()
        runtime.execute("TIMER SET cats 60", "Проверить файл.")
        self.assertTrue(runtime.execute("TIMER STOP cats", "").startswith("SYSTEM_OK"))
        self.assertTrue(runtime.execute("TIMER PERIOD cats 30", "").startswith("SYSTEM_OK"))
        self.assertTrue(runtime.execute("TIMER START cats", "").startswith("SYSTEM_OK"))
        self.assertTrue(runtime.execute("TIMER DELETE cats", "").startswith("SYSTEM_OK"))
        self.assertIn("not implemented", runtime.execute("GPIO WATCH 12", ""))


if __name__ == "__main__":
    unittest.main()
