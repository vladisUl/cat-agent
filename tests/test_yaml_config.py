from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml

from orchestration.config_env import main as config_env_main
from orchestration.yaml_config import PROJECT_ROOT, load_app_config


class YamlConfigTest(unittest.TestCase):
    def test_repository_config_uses_e4b_profile(self) -> None:
        config = load_app_config(PROJECT_ROOT / "cat-agent.yaml")
        self.assertEqual(config.litert.active_profile, "e4b")
        self.assertEqual(
            config.litert.active.model,
            "/storage/models/litertlm/gemma-4-E4B-it.litertlm",
        )
        self.assertEqual(config.litert.active.backend, "cpu")
        self.assertEqual(config.litert.active.cpu_threads, 8)
        self.assertEqual(config.runtime.http_timeout, 60)
        self.assertEqual(config.voice.wake_words, ("гена",))

    def test_active_profile_controls_litert_export(self) -> None:
        source = yaml.safe_load((PROJECT_ROOT / "cat-agent.yaml").read_text(encoding="utf-8"))
        source["litert"]["active_profile"] = "e2b"

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cat-agent.yaml"
            path.write_text(
                yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            output = io.StringIO()
            with mock.patch.dict(os.environ, {"CAT_AGENT_CONFIG": str(path)}, clear=False):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(config_env_main(["litert"]), 0)

        exported = output.getvalue()
        self.assertIn(
            "export LITERT_AGENT_MODEL_PATH=/storage/models/litertlm/gemma-4-E2B-it-gpu.litertlm",
            exported,
        )
        self.assertIn("export LITERT_AGENT_BACKEND=gpu", exported)
        self.assertIn("export LITERT_AGENT_ACTIVATION_DATA_TYPE=fp32", exported)
        self.assertIn("unset LITERT_AGENT_CPU_THREADS", exported)
        self.assertIn("export LITERT_AGENT_SPECULATIVE=1", exported)
        self.assertIn("export LITERT_AGENT_YNNPACK=0", exported)

    def test_unknown_option_is_rejected(self) -> None:
        source = yaml.safe_load((PROJECT_ROOT / "cat-agent.yaml").read_text(encoding="utf-8"))
        source["agent"]["manager_max_step"] = 99

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.yaml"
            path.write_text(
                yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Unknown agent option"):
                load_app_config(path)


if __name__ == "__main__":
    unittest.main()
