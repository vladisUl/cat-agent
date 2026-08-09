from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cat_agent.command_runtime import CommandRuntime


class CommandRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = CommandRuntime(
            self.root, ("shell",), max_file_bytes=1024, timeout_seconds=2
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_file_roundtrip(self) -> None:
        self.assertTrue(self.runtime.execute('echo "hello world" > note.txt').ok)
        read = self.runtime.execute("cat note.txt")
        self.assertEqual(read.stdout, "hello world\n")
        self.assertTrue(self.runtime.execute("mkdir data").ok)
        self.assertTrue(self.runtime.execute("mv note.txt data/note.txt").ok)
        self.assertTrue(self.runtime.execute("test -f data/note.txt").ok)

    def test_blocks_unassigned_command_and_escape(self) -> None:
        blocked = self.runtime.execute("mosquitto_sub -t test -C 1")
        self.assertEqual(blocked.exit_code, 126)
        escaped = self.runtime.execute("cat /etc/passwd")
        self.assertNotEqual(escaped.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
