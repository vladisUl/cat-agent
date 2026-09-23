from pathlib import Path
import sys
import tempfile
import unittest
from orchestration.process_runner import run_process


class ProcessRunnerTest(unittest.TestCase):
    def test_output_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_process(["/bin/bash", "-c", "yes x | head -c 100000"], cwd=temp, timeout=2, output_limit=64)
            self.assertEqual(result.returncode, 0)
            self.assertLess(len(result.stdout), 100)
            self.assertIn("truncated", result.stdout)

    def test_optional_stdin_is_delivered(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_process(
                ["/bin/cat"],
                cwd=temp,
                timeout=2,
                output_limit=1024,
                input_text='"привет" -f output.txt',
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, '"привет" -f output.txt')

    def test_large_stdin_does_not_deadlock_when_child_writes_first(self):
        with tempfile.TemporaryDirectory() as temp:
            code = (
                "import sys; "
                "sys.stdout.write('ready\\n'); sys.stdout.flush(); "
                "data=sys.stdin.read(); "
                "sys.stdout.write(str(len(data)))"
            )
            payload = "x" * 100000
            result = run_process(
                [sys.executable, "-c", code],
                cwd=temp,
                timeout=2,
                output_limit=1024,
                input_text=payload,
            )
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "ready\n100000")

    def test_timeout_kills_descendant_before_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            result = run_process(["/bin/bash", "-c", "(sleep 1; touch should-not-exist) & wait"], cwd=temp, timeout=0.1, output_limit=100)
            self.assertEqual(result.returncode, 124)
            self.assertIn("partial", result.stderr)
            # A barrier later than the descendant's intended write.
            run_process(["sleep", "1.1"], cwd=temp, timeout=2, output_limit=100)
            self.assertFalse((Path(temp) / "should-not-exist").exists())
