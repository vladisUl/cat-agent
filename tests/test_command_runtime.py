from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from cat_agent.workspace_command_runtime import CommandRuntime


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

    def test_safe_cat_append_requires_fresh_read(self) -> None:
        (self.root / "source.txt").write_text("Кот сидит на диване\n", encoding="utf-8")
        blocked = self.runtime.execute("cat source.txt >> user.txt")
        self.assertEqual(blocked.exit_code, 1)
        self.assertEqual(blocked.metadata.get("error_code"), "source_not_read")

        read = self.runtime.execute("cat source.txt")
        self.assertTrue(read.ok)
        appended = self.runtime.execute("cat source.txt >> user.txt")
        self.assertTrue(appended.ok)
        self.assertEqual(appended.operation, "append")
        self.assertEqual(
            (self.root / "user.txt").read_text(encoding="utf-8"),
            "Кот сидит на диване\n",
        )

    def test_safe_cat_append_rejects_changed_source(self) -> None:
        source = self.root / "source.txt"
        source.write_text("кот\n", encoding="utf-8")
        self.assertTrue(self.runtime.execute("cat source.txt").ok)
        source.write_text("собака\n", encoding="utf-8")
        changed = self.runtime.execute("cat source.txt >> user.txt")
        self.assertEqual(changed.exit_code, 1)
        self.assertEqual(changed.metadata.get("error_code"), "source_changed")

    def test_executes_workspace_executable_with_dot_slash(self) -> None:
        script = self.root / "timer_test.sh"
        script.write_text("#!/bin/sh\necho tick >> result.txt\n", encoding="utf-8")
        script.chmod(0o755)

        result = self.runtime.execute("./timer_test.sh")

        self.assertTrue(result.ok)
        self.assertEqual(result.operation, "exec")
        self.assertEqual(
            (self.root / "result.txt").read_text(encoding="utf-8"),
            "tick\n",
        )

    def test_rejects_workspace_file_without_executable_bit(self) -> None:
        script = self.root / "timer_test.sh"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o644)

        result = self.runtime.execute("./timer_test.sh")

        self.assertEqual(result.exit_code, 126)
        self.assertEqual(result.metadata.get("error_code"), "not_executable")

    def test_rejects_dot_slash_escape_outside_workspace(self) -> None:
        result = self.runtime.execute("./../outside.sh")
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
