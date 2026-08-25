from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

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

    def test_mqtt_sub_value_command_returns_only_selected_value(self) -> None:
        runtime = CommandRuntime(
            self.root, ("shell", "mqtt"), max_file_bytes=1024, timeout_seconds=2
        )
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="61.59\n", stderr=""
        )

        with patch(
            "orchestration.workspace_command_runtime.subprocess.run",
            return_value=completed,
        ) as run:
            result = runtime.execute(
                "mqtt_sub.sh zigbee2mqtt/temp_ulica humidity"
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.operation, "mqtt_sub")
        self.assertEqual(runtime.format_result(result), "61.59")
        self.assertEqual(result.metadata["topic"], "zigbee2mqtt/temp_ulica")
        self.assertEqual(result.metadata["field"], "humidity")
        self.assertEqual(
            run.call_args.args[0],
            [
                "/bin/bash",
                "-o",
                "pipefail",
                "-c",
                "mosquitto_sub -h 192.168.0.21 -p 1883 -t zigbee2mqtt/temp_ulica -C 1 -W 5 | jq -r '.humidity'",
            ],
        )

    def test_mqtt_pub_command_publishes_zigbee2mqtt_set_payload(self) -> None:
        runtime = CommandRuntime(
            self.root, ("mqtt",), max_file_bytes=1024, timeout_seconds=2
        )
        completed = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )

        with patch(
            "orchestration.workspace_command_runtime.subprocess.run",
            return_value=completed,
        ) as run:
            result = runtime.execute(
                "mqtt_pub.sh zigbee2mqtt/rozetka state=ON"
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.operation, "mqtt_pub")
        self.assertEqual(runtime.format_result(result), "OK")
        self.assertEqual(result.metadata["topic"], "zigbee2mqtt/rozetka")
        self.assertEqual(result.metadata["publish_topic"], "zigbee2mqtt/rozetka/set")
        self.assertEqual(result.metadata["field"], "state")
        self.assertEqual(result.metadata["value"], "ON")
        self.assertEqual(
            run.call_args.args[0],
            [
                "mosquitto_pub",
                "-h",
                "192.168.0.21",
                "-p",
                "1883",
                "-t",
                "zigbee2mqtt/rozetka/set",
                "-m",
                '{"state":"ON"}',
            ],
        )

    def test_mqtt_pub_requires_field_assignment(self) -> None:
        runtime = CommandRuntime(
            self.root, ("mqtt",), max_file_bytes=1024, timeout_seconds=2
        )

        with patch("orchestration.workspace_command_runtime.subprocess.run") as run:
            result = runtime.execute("mqtt_pub.sh zigbee2mqtt/rozetka ON")

        self.assertFalse(result.ok)
        self.assertEqual(result.operation, "mqtt_pub")
        self.assertEqual(result.metadata.get("error_code"), "invalid_assignment")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
