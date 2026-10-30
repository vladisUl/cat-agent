"""Real file-append semantics without a running MQTT broker or model."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from litert_agent_tools.config import MqttConfig
from litert_agent_tools.record_tools import (
    create_record_outdoor_weather_tool, extract_record_filename,
)
from litert_agent_tools.shell_tools import ShellGate
from litert_agent_tools.tool_audit import capture_task_tools


class WeatherRecordingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.gate = ShellGate(enabled=True)
        self.tool = create_record_outdoor_weather_tool(
            MqttConfig(), self.root, gate=self.gate,
        )

    def test_create_then_append_each_measurement_with_receipt(self):
        with patch(
            "litert_agent_tools.record_tools._read_measurement",
            side_effect=[
                {"ok": True, "temperature_c": 10.5,
                 "humidity_percent": 76.5},
                {"ok": True, "temperature_c": 10.7,
                 "humidity_percent": 77.1},
            ],
        ) as mqtt:
            with capture_task_tools() as audit:
                first = self.tool("temp.txt")
                self.assertTrue(first["ok"], first)
                self.assertEqual(first["file_path"], str(self.root / "temp.txt"))
                self.assertGreater(first["bytes_written"], 0)
                self.assertEqual(audit.successful("record_outdoor_weather"), first)
                second = self.tool("temp.txt")
                self.assertTrue(second["ok"], second)
            self.assertEqual(mqtt.call_count, 2)
        rows = (self.root / "temp.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn("10.50 C 76.50 %", rows[0])
        self.assertIn("10.70 C 77.10 %", rows[1])
        self.assertEqual(second["file_size_bytes"],
                         (self.root / "temp.txt").stat().st_size)

    def test_mqtt_failure_does_not_create_file(self):
        with patch(
            "litert_agent_tools.record_tools._read_measurement",
            return_value={"ok": False, "error": "MQTT timeout"},
        ):
            result = self.tool("temp.txt")
        self.assertFalse(result["ok"])
        self.assertFalse((self.root / "temp.txt").exists())

    def test_warmup_and_unsafe_paths_never_write(self):
        self.gate.enabled = False
        with patch("litert_agent_tools.record_tools._read_measurement") as mqtt:
            result = self.tool("temp.txt")
            self.assertFalse(result["ok"])
            self.assertFalse(mqtt.called)
        self.gate.enabled = True
        for bad in ("../temp.txt", "/tmp/temp.txt", "sub/temp.txt",
                    "bad.txt/other", ".txt", "temp.csv", "a..txt"):
            with self.subTest(bad=bad):
                self.assertFalse(self.tool(bad)["ok"])
        self.assertFalse((self.root / "temp.txt").exists())

    def test_symlink_destination_is_rejected_without_touching_target(self):
        other = self.root / "target.txt"
        other.write_text("unchanged\n", encoding="utf-8")
        (self.root / "temp.txt").symlink_to(other)
        with patch(
            "litert_agent_tools.record_tools._read_measurement",
            return_value={"ok": True, "temperature_c": 10,
                          "humidity_percent": 70},
        ):
            result = self.tool("temp.txt")
        self.assertFalse(result["ok"])
        self.assertEqual(other.read_text(encoding="utf-8"), "unchanged\n")

    def test_filename_extraction_rejects_ambiguous_and_paths(self):
        self.assertEqual(
            extract_record_filename("записать в файл temp.txt"), "temp.txt",
        )
        self.assertIsNone(extract_record_filename("записать в /tmp/temp.txt"))
        self.assertIsNone(extract_record_filename(
            "записать в temp.txt и copy.txt",
        ))


if __name__ == "__main__":
    unittest.main()
