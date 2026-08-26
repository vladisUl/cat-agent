from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.event_store import EventStore
from orchestration.mqtt_events import MqttEventMonitor, MqttTopicCatalog


class MqttEventTest(unittest.TestCase):
    def test_catalog_parses_discrete_boolean_value_semantics(self) -> None:
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

    def test_pir_emits_each_change_and_suppresses_repeated_state(self) -> None:
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
                'zigbee2mqtt/dvigen_verh {"occupancy":false,"linkquality":240}'
            )
            self.assertEqual(emitted, [])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":true,"linkquality":240}'
            )
            self.assertEqual(emitted, [("task_mqtt1", "true")])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":true,"linkquality":193}'
            )
            self.assertEqual(emitted, [("task_mqtt1", "true")])

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":false,"linkquality":193}'
            )
            self.assertEqual(
                emitted,
                [("task_mqtt1", "true"), ("task_mqtt1", "false")],
            )

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":false,"linkquality":120}'
            )
            self.assertEqual(
                emitted,
                [("task_mqtt1", "true"), ("task_mqtt1", "false")],
            )

            monitor._consume_line(
                'zigbee2mqtt/dvigen_verh {"occupancy":true,"linkquality":120}'
            )
            self.assertEqual(
                emitted,
                [
                    ("task_mqtt1", "true"),
                    ("task_mqtt1", "false"),
                    ("task_mqtt1", "true"),
                ],
            )

    def test_non_boolean_discrete_field_uses_same_change_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = EventStore(Path(temp) / "events.json")
            store.register(
                1,
                "свет",
                source="mqtt",
                topic="zigbee2mqtt/light",
                field="state",
                value_type="string",
                values=("ON", "OFF"),
                command="mqtt_sub.sh zigbee2mqtt/light state",
            )
            emitted: list[str] = []
            monitor = MqttEventMonitor(
                store,
                lambda _binding, value: emitted.append(value),
            )

            monitor._consume_line('zigbee2mqtt/light {"state":"OFF"}')
            self.assertEqual(emitted, [])
            monitor._consume_line('zigbee2mqtt/light {"state":"OFF"}')
            self.assertEqual(emitted, [])
            monitor._consume_line('zigbee2mqtt/light {"state":"ON"}')
            self.assertEqual(emitted, ["ON"])
            monitor._consume_line('zigbee2mqtt/light {"state":"ON"}')
            self.assertEqual(emitted, ["ON"])
            monitor._consume_line('zigbee2mqtt/light {"state":"OFF"}')
            self.assertEqual(emitted, ["ON", "OFF"])


if __name__ == "__main__":
    unittest.main()
