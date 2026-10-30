"""User-defined native LiteRT Tool adapter tests (no model or LiteRT install)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from litert_agent_tools.conversation import ToolExecutionLedger
from litert_agent_tools.external_tools import load_external_command_tools, native_tool_name
from litert_agent_tools.shell_tools import ShellConfig, ShellGate


class ExternalCommandToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.gate = ShellGate()
        self.fake_litert = ModuleType("litert_lm")
        self.fake_litert.Tool = type("Tool", (), {})

    def write_tool(self, name="get_current_time", command=None, description="Current time"):
        if command is None:
            command = ["/usr/bin/date", "+%Y-%m-%d %H:%M:%S %z"]
        path = self.directory / f"{name}.json"
        path.write_text(json.dumps({
            "name": name, "description": description, "command": command,
        }), encoding="utf-8")
        return path

    def load(self):
        with patch.dict(sys.modules, {"litert_lm": self.fake_litert}):
            return load_external_command_tools(
                self.directory, ShellConfig(self.directory), gate=self.gate,
            )

    def test_repo_date_definition_is_registered_with_native_schema(self):
        repo_path = Path(__file__).resolve().parents[1] / "config/native_tools/get_current_time.json"
        (self.directory / "date.json").write_text(repo_path.read_text(encoding="utf-8"), encoding="utf-8")
        tools = self.load()
        self.assertEqual(len(tools), 1)
        self.assertIsInstance(tools[0], self.fake_litert.Tool)
        self.assertEqual(native_tool_name(tools[0]), "get_current_time")
        function = tools[0].get_tool_description()["function"]
        self.assertEqual(function["parameters"]["properties"], {})
        self.assertEqual(function["parameters"]["required"], [])
        self.assertNotIn("command", str(function))

    def test_real_execution_uses_fixed_command_and_warm_gate(self):
        self.write_tool()
        tool, = self.load()
        self.assertFalse(tool.execute({})["ok"])
        self.gate.enabled = True
        with patch("litert_agent_tools.external_tools._execute",
                   return_value={"ok": True, "exit_code": 0, "output": "2026-10-30 18:30:00 +0500"}) as runner:
            result = tool.execute({})
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "2026-10-30 18:30:00 +0500")
        self.assertEqual(runner.call_args.args[1], "/usr/bin/date '+%Y-%m-%d %H:%M:%S %z'")
        self.assertFalse(tool.execute({"command": "rm -rf /"})["ok"])

    def test_ledger_tracks_tool_object_success_and_failure(self):
        self.write_tool()
        tool, = self.load()
        ledger = ToolExecutionLedger()
        with patch.dict(sys.modules, {"litert_lm": self.fake_litert}):
            measured = ledger.wrap(tool)
        self.assertIsInstance(measured, self.fake_litert.Tool)
        self.assertEqual(native_tool_name(measured), "get_current_time")
        ledger.active = True
        self.assertFalse(measured.execute({})["ok"])
        self.assertEqual((ledger.calls, ledger.status_code), (1, 1))
        ledger.active = False
        ledger.calls = 0
        ledger.status_code = 0
        ledger.active = True
        self.gate.enabled = True
        with patch("litert_agent_tools.external_tools._execute",
                   return_value={"ok": True, "exit_code": 0, "output": "OK"}):
            self.assertTrue(measured.execute({})["ok"])
        self.assertEqual((ledger.calls, ledger.status_code), (1, 0))

    def test_rejects_bad_and_ambiguous_descriptions(self):
        self.write_tool(name="bad-name")
        with self.assertRaisesRegex(ValueError, "invalid tool name"):
            self.load()
        (self.directory / "bad-name.json").unlink()

        self.write_tool(command=["date"])
        with self.assertRaisesRegex(ValueError, "must be absolute"):
            self.load()
        (self.directory / "get_current_time.json").unlink()

        self.write_tool()
        (self.directory / "other.json").write_text(
            (self.directory / "get_current_time.json").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate tool name"):
            self.load()


if __name__ == "__main__":
    unittest.main()
