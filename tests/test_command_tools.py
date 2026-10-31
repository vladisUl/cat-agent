"""Native LiteRT JSON command tool factory: no model installation required."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

from litert_agent_tools.command_tools import create_command_tools
from litert_agent_tools.conversation import ToolExecutionLedger
from litert_agent_tools.shell_tools import ShellGate


class CommandToolTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.specs = self.root / "command_tools.json"
        self.gate = ShellGate()
        self._write([{
            "name": "get_cat_water_level",
            "description": "Получить уровень воды у кота.",
            "command": "cat_water",
            "parameters": {
                "room": {"type": "string", "enum": ["kitchen", "bedroom"],
                         "description": "Комната"}
            },
        }, {
            "name": "get_cat_food_level",
            "description": "Получить уровень корма.",
            "command": "cat_food",
        }])
        self._executable("cat_water", "import sys\nprint('70%')\n")
        self._executable("cat_food", "print('{\"level_percent\": 62}')\n")

    def _write(self, tools):
        self.specs.write_text(json.dumps({"tools": tools}, ensure_ascii=False),
                              encoding="utf-8")

    def _executable(self, name, body):
        executable = self.root / name
        executable.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
        executable.chmod(0o755)
        return executable

    def _load(self, **kwargs):
        return create_command_tools(self.root, self.specs, gate=self.gate, **kwargs)

    def test_real_typed_python_signatures_and_docstrings(self):
        tools = self._load()
        self.assertEqual([tool.__name__ for tool in tools],
                         ["get_cat_water_level", "get_cat_food_level"])
        self.assertEqual(str(inspect.signature(tools[0])), "(room: str) -> dict")
        self.assertEqual(str(inspect.signature(tools[1])), "() -> dict")
        self.assertIn("kitchen", tools[0].__doc__)
        self.assertIn("Комната", tools[0].__doc__)

    def test_warm_gate_and_plain_text_stdout(self):
        water, _ = self._load()
        self.assertEqual(water("kitchen")["ok"], False)
        self.gate.enabled = True
        self.assertEqual(water(room="kitchen"), {"ok": True, "result": "70%"})
        self.assertFalse(water("garage")["ok"])
        self.assertFalse(water(17)["ok"])

    def test_no_parameters_and_json_stdout(self):
        _, food = self._load()
        self.gate.enabled = True
        self.assertEqual(food(), {"ok": True, "result": {"level_percent": 62}})

    def test_argv_passes_separate_values_without_shell(self):
        self._executable("cat_water", "import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
        self._write([{
            "name": "get_cat_water_level", "description": "Уровень",
            "command": "cat_water", "parameters": {
                "room": {"type": "string", "description": "Комната"},
                "threshold": {"type": "integer", "description": "Порог"},
                "enabled": {"type": "boolean", "description": "Использовать"}
            }
        }])
        water, = self._load()
        self.gate.enabled = True
        self.assertEqual(water("kitchen; echo danger", 5, True),
                         {"ok": True, "result": ["kitchen; echo danger", "5", "true"]})

    def test_nonzero_exit_and_bounded_stderr(self):
        self._executable("cat_food", "import sys\nsys.stderr.write('sensor_timeout')\nsys.exit(2)\n")
        self.gate.enabled = True
        _, food = self._load()
        self.assertEqual(food(), {"ok": False, "exit_code": 2, "error": "sensor_timeout"})

    def test_timeout_and_output_limit(self):
        self._executable("cat_food", "import time\ntime.sleep(1)\nprint('late')\n")
        self.gate.enabled = True
        _, food = self._load(timeout_seconds=0.05)
        self.assertEqual(food()["error"], "timeout")
        self._executable("cat_food", "print('X' * 1000)\n")
        _, food = self._load(max_output_bytes=16)
        self.assertEqual(food()["error"], "output_limit_exceeded")

    def test_ledger_tracks_tool_call_without_changing_signature(self):
        self.gate.enabled = True
        water, = self._load()[0:1]
        ledger = ToolExecutionLedger()
        measured = ledger.wrap(water)
        self.assertEqual(inspect.signature(measured), inspect.signature(water))
        ledger.active = True
        self.assertEqual(measured("kitchen"), {"ok": True, "result": "70%"})
        self.assertEqual((ledger.calls, ledger.status_code), (1, 0))

    def test_unknown_and_duplicate_definitions_rejected(self):
        good = {"name": "tool", "description": "Valid", "command": "cat_food"}
        for broken in (
            [good, good],
            [dict(good, command="../cat_food")],
            [dict(good, name="x-y")],
            [dict(good, parameters={"room": {"type": "list", "description": "Bad"}})],
            [dict(good, parameters={"room": {"type": "string", "description": "Room",
                                            "enum": ["kitchen", "kitchen"]}})],
            [dict(good, parameters={"class": {"type": "string", "description": "Bad"}})],
        ):
            with self.subTest(broken=broken):
                self._write(broken)
                with self.assertRaises(ValueError):
                    self._load()

    def test_missing_program_and_directory_escape_rejected(self):
        self._write([{"name": "bad", "description": "bad", "command": "missing"}])
        with self.assertRaisesRegex(ValueError, "executable"):
            self._load()
        (self.root / "outside").symlink_to("/usr/bin/env")
        self._write([{"name": "outside", "description": "bad", "command": "outside"}])
        with self.assertRaisesRegex(ValueError, "outside command directory"):
            self._load()

    def test_empty_configuration_does_not_require_installing_executable_directory(self):
        self._write([])
        self.assertEqual(create_command_tools(self.root / "not-created", self.specs), [])


if __name__ == "__main__":
    unittest.main()
