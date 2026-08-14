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


if __name__ == "__main__":
    unittest.main()
