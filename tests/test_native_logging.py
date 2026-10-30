"""Native LiteRT CORE log file is independent from OpenAI and contains transcript text."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import logging
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from agent_core_tools.native_logging import DEFAULT_LOG_PATH, configure_native_logging


class NativeLoggingTests(unittest.TestCase):
    def test_file_stderr_and_world_readable_permissions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "logs" / "litert.log"
            stream = io.StringIO()
            isolated_root = logging.Logger("native-test", level=logging.NOTSET)
            with redirect_stderr(stream):
                with patch("agent_core_tools.native_logging.logging.getLogger",
                           return_value=isolated_root):
                    actual = configure_native_logging("INFO", path=path)
            try:
                isolated_root.info("USER -> MANAGER %r", "привет")
                isolated_root.info("MANAGER -> USER %r", "ответ")
                isolated_root.debug("hidden debug event")
                for handler in isolated_root.handlers:
                    handler.flush()
                text = path.read_text(encoding="utf-8")
                self.assertEqual(actual, path)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                self.assertIn("USER -> MANAGER 'привет'", text)
                self.assertIn("MANAGER -> USER 'ответ'", text)
                self.assertNotIn("hidden debug event", text)
                self.assertIn("USER -> MANAGER", stream.getvalue())
                self.assertNotEqual(path, DEFAULT_LOG_PATH.parent / "cat-agent.log")
            finally:
                for handler in list(isolated_root.handlers):
                    isolated_root.removeHandler(handler)
                    handler.close()

    def test_env_override_and_level(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "litert.log"
            isolated_root = logging.Logger("native-test", level=logging.NOTSET)
            with patch.dict(os.environ, {"CAT_AGENT_LITERT_LOG_PATH": str(path)}):
                with patch("agent_core_tools.native_logging.logging.getLogger",
                           return_value=isolated_root):
                    actual = configure_native_logging("ERROR")
            try:
                self.assertEqual(actual, path)
                isolated_root.warning("warn")
                isolated_root.error("error")
                for handler in isolated_root.handlers:
                    handler.flush()
                output = path.read_text(encoding="utf-8")
                self.assertIn("error", output)
                self.assertNotIn("warn", output)
            finally:
                for handler in list(isolated_root.handlers):
                    isolated_root.removeHandler(handler)
                    handler.close()

    def test_rejects_relative_log_path(self):
        with self.assertRaisesRegex(ValueError, "absolute"):
            configure_native_logging("INFO", path=Path("relative.log"))


if __name__ == "__main__":
    unittest.main()
