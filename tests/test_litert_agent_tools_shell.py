"""Native shell tool tests (without LiteRT, model, MQTT, or live CORE)."""

from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import time
import unittest

from litert_agent_tools.shell_tools import (
    DEFAULT_SHELL_CWD, MAX_COMMAND_CHARS, ShellConfig, ShellGate, create_shell_tool,
)


class ShellToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _tool(self, *, timeout=5.0, limit=16384, gate=None):
        return create_shell_tool(
            ShellConfig(self.root, timeout_seconds=timeout, max_output_bytes=limit),
            gate=gate,
        )

    def test_callable_schema_and_real_bash_pipeline(self):
        shell = self._tool()
        self.assertEqual(shell.__name__, "shell")
        signature = inspect.signature(shell)
        self.assertEqual(list(signature.parameters), ["command"])
        self.assertIn("Args:", inspect.getdoc(shell))
        result = shell("printf 'a\\nb\\n' | tail -n 1")
        self.assertEqual(result, {"ok": True, "exit_code": 0, "output": "b\n"})

    def test_default_shell_directory(self):
        self.assertEqual(DEFAULT_SHELL_CWD, Path("/opt/model"))

    def test_runs_from_configured_directory(self):
        result = self._tool()("pwd")
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"].strip(), str(self.root.resolve()))

    def test_reports_nonzero_exit_and_stderr(self):
        result = self._tool()("printf 'problem' >&2; exit 17")
        self.assertFalse(result["ok"])
        self.assertEqual(result["exit_code"], 17)
        self.assertEqual(result["error"], "exit_code_17")
        self.assertEqual(result["output"], "problem")

    def test_warm_gate_denies_until_explicitly_enabled(self):
        gate = ShellGate()
        shell = self._tool(gate=gate)
        result = shell("touch should_not_exist.txt")
        self.assertFalse(result["ok"])
        self.assertIn("warm-up", result["error"])
        self.assertFalse((self.root / "should_not_exist.txt").exists())
        gate.enabled = True
        result = shell("touch now_exists.txt")
        self.assertTrue(result["ok"])
        self.assertTrue((self.root / "now_exists.txt").exists())

    def test_timeout_kills_process_group(self):
        started = time.monotonic()
        result = self._tool(timeout=0.35)("sleep 5")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout")
        self.assertIsNone(result["exit_code"])
        self.assertLess(time.monotonic() - started, 3.0)

    def test_output_bound_terminates_command_not_only_truncates_response(self):
        result = self._tool(limit=4)("printf '12345678'")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "output_limit_exceeded")
        self.assertEqual(result["output"], "1234")
        self.assertIsNone(result["exit_code"])

    def test_invalid_commands_do_not_launch(self):
        shell = self._tool()
        for command in ("", "  ", None, 42, "x" * (MAX_COMMAND_CHARS + 1),
                        "a\x00b"):
            with self.subTest(command=repr(command)[:40]):
                self.assertFalse(shell(command)["ok"])

    def test_config_bounds_and_path_validation(self):
        cases = [
            {"workdir": self.root / "not-a-directory"},
            {"workdir": self.root, "timeout_seconds": 0},
            {"workdir": self.root, "timeout_seconds": 121},
            {"workdir": self.root, "max_output_bytes": 0},
            {"workdir": self.root, "max_output_bytes": 65537},
        ]
        for config in cases:
            with self.subTest(config=config), self.assertRaises(ValueError):
                ShellConfig(**config)

    def test_result_log_does_not_duplicate_full_command_output(self):
        shell = self._tool()
        with self.assertLogs("litert_agent_tools.shell_tools", level="INFO") as logs:
            result = shell("printf 'sample_result_text'")
        self.assertTrue(result["ok"])
        self.assertIn("native-tool CALL name=shell", "\n".join(logs.output))
        self.assertIn("native-tool RESULT name=shell", "\n".join(logs.output))
        self.assertNotIn("sample_result_text", logs.output[-1])


if __name__ == "__main__":
    unittest.main()
