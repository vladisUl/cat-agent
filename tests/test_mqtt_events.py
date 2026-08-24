from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.event_store import EventStore
from orchestration.mqtt_events import MqttEventMonitor, MqttTopicCatalog


class MqttEventTest(unittest.TestCase):
    def test_catalog_parses_one_and_two_significant_boolean_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mqtt.txt"
            path.write_text(
                "topics:\n"
                "zigbee2mqtt/pir: движение; occupancy: boolean; "
                "(true: движение обнаружено)\n"
                "zigbee2mqtt/door: дверь; contact: boolean; "
                "(true: дверь закрыта, false: дверь открыта)\n",
                encoding="utf-8",
            )

            catalog = MqttTopicCatalog(path)

            pir = catalog.require("zigbee2mqtt/pir", "occupancy")
            self.assertEqual(pir.value_type, "boolean")
            self.assertEqual(pir.values, ("true",))

            door = catalog.require("zigbee2mqtt/door", "contact")
            self.assertEqual(door.value_type, "boolean")
            self.assertEqual(door.values, ("true", "false"))

    def test_pir_false_rearms_but_only_true_emits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EventStore(Path(temp) / "events.json")
            store.register(
                1,
                "движение в коридоре",
                source="mqtt",
                topic="zigbee2mqtt/dvigen_verh",
                field="occupancy",
                value_type="boolean",
                values=("true",),
                command="mqtt_sub.sh zigbee2mqtt/dvigen_verh occupancy",
            )
            emitted: list[tuple[str, str]] = []
            monitor = MqttEventMonitor(
                store,
                lambda binding, value: emitted.append((binding.name, value)),
            )

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":false}'
            )
            self.assertEqual(emitted, [])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":true}'
            )
            self.assertEqual(emitted, [("task_mqtt1", "true")])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":true}'
            )
            self.assertEqual(emitted, [("task_mqtt1", "true")])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":false}'
            )
            self.assertEqual(emitted, [("task_mqtt1", "true")])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":true}'
            )
            self.assertEqual(
                emitted,
                [("task_mqtt1", "true"), ("task_mqtt1", "true")],
            )

    def test_boolean_with_two_values_emits_both_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EventStore(Path(temp) / "events.json")
            store.register(
                1,
                "дверь",
                source="mqtt",
                topic="zigbee2mqtt/door",
                field="contact",
                value_type="boolean",
                values=("true", "false"),
                command="mqtt_sub.sh zigbee2mqtt/door contact",
            )
            emitted: list[str] = []
            monitor = MqttEventMonitor(
                store,
                lambda _binding, value: emitted.append(value),
            )

            monitor._consume_line('zigbee2mqtt/door {"contact":true}')
            self.assertEqual(emitted, [])
            monitor._consume_line('zigbee2mqtt/door {"contact":false}')
            self.assertEqual(emitted, ["false"])
            monitor._consume_line('zigbee2mqtt/door {"contact":true}')
            self.assertEqual(emitted, ["false", "true"])


if __name__ == "__main__":
    unittest.main()
