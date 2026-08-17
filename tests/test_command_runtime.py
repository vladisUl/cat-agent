from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from orchestration.workspace_command_runtime import CommandRuntime


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

    def test_shell_starts_in_workspace(self) -> None:
        result = self.runtime.execute("pwd")
        self.assertTrue(result.ok)
        self.assertEqual(Path(result.stdout.strip()), self.root.resolve())

    def test_boolean_shell_chain_used_by_agent(self) -> None:
        (self.root / "user.txt").write_text("ok\n", encoding="utf-8")

        present = self.runtime.execute(
            'test -f user.txt && echo "ОК" || echo "Авария"'
        )
        self.assertTrue(present.ok)
        self.assertEqual(present.stdout, "ОК\n")

        (self.root / "user.txt").unlink()
        absent = self.runtime.execute(
            'test -f user.txt && echo "ОК" || echo "Авария"'
        )
        self.assertTrue(absent.ok)
        self.assertEqual(absent.stdout, "Авария\n")

    def test_pipeline_redirection_and_command_substitution(self) -> None:
        result = self.runtime.execute(
            'printf "one\\ntwo\\n" | wc -l > count.txt && echo "$(cat count.txt)"'
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout.strip(), "2")
        self.assertEqual((self.root / "count.txt").read_text().strip(), "2")

    def test_executes_workspace_executable_with_dot_slash(self) -> None:
        script = self.root / "timer_test.sh"
        script.write_text("#!/bin/sh\necho tick >> result.txt\n", encoding="utf-8")
        script.chmod(0o755)

        result = self.runtime.execute("./timer_test.sh")

        self.assertTrue(result.ok)
        self.assertEqual(result.operation, "bash")
        self.assertEqual(
            (self.root / "result.txt").read_text(encoding="utf-8"),
            "tick\n",
        )

    def test_timeout_is_kept(self) -> None:
        runtime = CommandRuntime(
            self.root, ("shell",), max_file_bytes=1024, timeout_seconds=1
        )
        result = runtime.execute("sleep 5")
        self.assertEqual(result.exit_code, 124)
        self.assertEqual(result.metadata.get("error_code"), "timeout")

    def test_runtime_without_shell_skill_stays_restricted(self) -> None:
        runtime = CommandRuntime(
            self.root, ("mqtt",), max_file_bytes=1024, timeout_seconds=2
        )
        blocked = runtime.execute("ls")
        self.assertEqual(blocked.exit_code, 126)
        self.assertEqual(blocked.metadata.get("error_code"), "command_not_permitted")


if __name__ == "__main__":
    unittest.main()
