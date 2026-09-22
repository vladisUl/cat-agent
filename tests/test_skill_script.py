from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from orchestration.skill_script import SKILL_SILENT, run_skill_script
from orchestration.workspace_command_runtime import CommandRuntime


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/a9sAAAAASUVORK5CYII="
)


class SkillScriptTest(unittest.TestCase):
    def runtime(self, root: Path, *skills: str) -> CommandRuntime:
        return CommandRuntime(
            root,
            tuple(skills),
            max_file_bytes=4096,
            timeout_seconds=2,
        )

    def test_assigned_skill_runs_external_then_attaches_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data" / "camera").mkdir(parents=True)
            (root / "seed.png").write_bytes(PNG)

            command = root / "web_shot"
            command.write_text(
                "#!/bin/sh\ncp seed.png \"$1\"\n",
                encoding="utf-8",
            )
            command.chmod(0o755)

            (root / "camera.sh").write_text(
                "web_shot /camera/test.png\n"
                "read_pic.sh /camera/test.png\n",
                encoding="utf-8",
            )

            result = run_skill_script(
                "camera.sh",
                self.runtime(root, "camera"),
                SimpleNamespace(supports_images=True),
            )

            self.assertIsInstance(result, list)
            assert isinstance(result, list)
            self.assertEqual((root / "data" / "camera" / "test.png").read_bytes(), PNG)
            self.assertEqual(result[1]["type"], "image_url")

    def test_logical_absolute_argument_is_rewritten_under_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data" / "skill").mkdir(parents=True)

            command = root / "record_path"
            command.write_text(
                "#!/bin/sh\nprintf '%s' \"$1\" > received.txt\n",
                encoding="utf-8",
            )
            command.chmod(0o755)
            (root / "demo.sh").write_text(
                "record_path skill/test.png\n",
                encoding="utf-8",
            )

            result = run_skill_script(
                "demo.sh",
                self.runtime(root, "demo"),
                SimpleNamespace(supports_images=True),
            )

            self.assertIs(result, SKILL_SILENT)
            self.assertEqual(
                (root / "received.txt").read_text(encoding="utf-8"),
                str(root.resolve() / "data" / "skill" / "test.png"),
            )

    def test_last_external_stdout_is_skill_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()

            first = root / "first"
            first.write_text("#!/bin/sh\nprintf 'FIRST\\n'\n", encoding="utf-8")
            first.chmod(0o755)

            second = root / "second"
            second.write_text("#!/bin/sh\nprintf 'SECOND\\n'\n", encoding="utf-8")
            second.chmod(0o755)

            (root / "demo.sh").write_text("first\nsecond\n", encoding="utf-8")

            result = run_skill_script(
                "demo.sh",
                self.runtime(root, "demo"),
                SimpleNamespace(supports_images=True),
            )

            self.assertEqual(result, "SECOND")

    def test_empty_stdout_of_last_external_command_means_silent_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()

            first = root / "first"
            first.write_text("#!/bin/sh\nprintf 'FIRST\\n'\n", encoding="utf-8")
            first.chmod(0o755)

            second = root / "second"
            second.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            second.chmod(0o755)

            (root / "demo.sh").write_text("first\nsecond\n", encoding="utf-8")

            result = run_skill_script(
                "demo.sh",
                self.runtime(root, "demo"),
                SimpleNamespace(supports_images=True),
            )

            self.assertIs(result, SKILL_SILENT)

    def test_external_command_must_exist_and_be_executable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "demo.sh").write_text("missing /test.png\n", encoding="utf-8")
            result = run_skill_script(
                "demo.sh",
                self.runtime(root, "demo"),
                SimpleNamespace(supports_images=True),
            )
            self.assertIn("command not found", result)

            command = root / "missing"
            command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            result = run_skill_script(
                "demo.sh",
                self.runtime(root, "demo"),
                SimpleNamespace(supports_images=True),
            )
            self.assertIn("file is not executable", result)

    def test_only_assigned_skill_name_is_interpreted_as_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "camera.sh").write_text("anything\n", encoding="utf-8")

            result = run_skill_script(
                "camera.sh",
                self.runtime(root, "shell"),
                SimpleNamespace(supports_images=True),
            )

            self.assertIsNone(result)


    def test_internal_mqtt_command_is_available_inside_dynamic_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            (root / "demo.sh").write_text(
                "mqtt_sub.sh zigbee2mqtt/temp temperature\n",
                encoding="utf-8",
            )
            completed = __import__("subprocess").CompletedProcess(
                args=[], returncode=0, stdout="21.5\n", stderr=""
            )

            with patch(
                "orchestration.workspace_command_runtime.run_process",
                return_value=completed,
            ):
                result = run_skill_script(
                    "demo.sh",
                    self.runtime(root, "demo"),
                    SimpleNamespace(supports_images=True),
                )

            self.assertIs(result, SKILL_SILENT)

    def test_parent_traversal_in_logical_data_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()

            command = root / "writer"
            command.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            command.chmod(0o755)
            (root / "demo.sh").write_text(
                "writer /../escape.png\n",
                encoding="utf-8",
            )

            result = run_skill_script(
                "demo.sh",
                self.runtime(root, "demo"),
                SimpleNamespace(supports_images=True),
            )

            self.assertIn("invalid data path", result)


if __name__ == "__main__":
    unittest.main()
