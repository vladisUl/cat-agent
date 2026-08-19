from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from orchestration.event_store import EventStore


class EventStoreTest(unittest.TestCase):
    def test_register_persists_expected_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.json"
            store = EventStore(path)

            binding = store.register(
                1,
                "датчик открытия двери",
                source="gpio",
            )

            self.assertEqual(binding.name, "task_gpio1")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                payload,
                {
                    "task_gpio1": {
                        "source": "gpio",
                        "task_id": 1,
                        "description": "датчик открытия двери",
                    }
                },
            )

            restored = EventStore(path)
            resolved = restored.resolve("gpio", "task_gpio1")
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved.task_id, 1)  # type: ignore[union-attr]

    def test_unregister_task_removes_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.json"
            store = EventStore(path)
            store.register(1, "door", source="gpio")

            store.unregister_task(1)

            self.assertIsNone(store.resolve("gpio", "task_gpio1"))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {})


if __name__ == "__main__":
    unittest.main()
