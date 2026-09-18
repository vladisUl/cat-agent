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
    def _capture(self, target: str, *, config_path: Path | None = None) -> str:
        output = io.StringIO()
        patch = {}
        if config_path is not None:
            patch["CAT_AGENT_CONFIG"] = str(config_path)
        with mock.patch.dict(os.environ, patch, clear=False):
            with contextlib.redirect_stdout(output):
                self.assertEqual(config_env_main([target]), 0)
        return output.getvalue()

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
        self.assertEqual(config.openai.active_profile, "laptop-12b")
        self.assertEqual(config.openai.active.base_url, "http://192.168.0.129:8082/v1")
        self.assertEqual(config.openai.active.model, "gemma-4-12b")
        self.assertEqual(config.openai.active.readiness, "openai")

    def test_active_profile_controls_litert_export(self) -> None:
        source = yaml.safe_load((PROJECT_ROOT / "cat-agent.yaml").read_text(encoding="utf-8"))
        source["litert"]["active_profile"] = "e2b"

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cat-agent.yaml"
            path.write_text(
                yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            exported = self._capture("litert", config_path=path)

        self.assertIn(
            "export LITERT_AGENT_MODEL_PATH=/storage/models/litertlm/gemma-4-E2B-it-gpu.litertlm",
            exported,
        )
        self.assertIn("export LITERT_AGENT_BACKEND=gpu", exported)
        self.assertIn("export LITERT_AGENT_ACTIVATION_DATA_TYPE=fp32", exported)
        self.assertIn("unset LITERT_AGENT_CPU_THREADS", exported)
        self.assertIn("export LITERT_AGENT_SPECULATIVE=1", exported)
        self.assertIn("export LITERT_AGENT_YNNPACK=0", exported)

    def test_active_profile_controls_openai_export(self) -> None:
        source = yaml.safe_load((PROJECT_ROOT / "cat-agent.yaml").read_text(encoding="utf-8"))
        source["openai"]["active_profile"] = "cloud"

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cat-agent.yaml"
            path.write_text(
                yaml.safe_dump(source, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            exported = self._capture("openai", config_path=path)

        self.assertIn("export CAT_AGENT_API_BASE_URL=http://127.0.0.1:11434/v1", exported)
        self.assertIn("export CAT_AGENT_MODEL=gemma4:31b-cloud", exported)
        self.assertIn("export CAT_AGENT_OPENAI_READINESS=ollama", exported)

    def test_openai_and_common_settings_are_exported(self) -> None:
        exported = self._capture("openai")
        expected = (
            "export CAT_AGENT_API_BASE_URL=http://192.168.0.129:8082/v1",
            "export CAT_AGENT_MODEL=gemma-4-12b",
            "export CAT_AGENT_OPENAI_READINESS=openai",
            "export CAT_AGENT_REASONING_EFFORT=none",
            "export CAT_AGENT_AGENT_COUNT=3",
            "export CAT_AGENT_MAX_MANAGER_STEPS=12",
            "export CAT_AGENT_MAX_AGENT_STEPS=12",
            "export CAT_AGENT_MANAGER_MAX_OUTPUT_TOKENS=256",
            "export CAT_AGENT_AGENT_MAX_OUTPUT_TOKENS=128",
            "export CAT_AGENT_TEMPERATURE=0.0",
            "export CAT_AGENT_TOP_P=1.0",
            "export CAT_AGENT_WORKSPACE=/opt/model",
            "export CAT_AGENT_PROMPT_DIR=/opt/cat-agent/prompts",
            "export CAT_AGENT_COMMAND_TIMEOUT_SECONDS=20",
            "export CAT_AGENT_HTTP_TIMEOUT_SECONDS=60",
            "export CAT_AGENT_REQUEST_RETRIES=0",
            "export CAT_AGENT_RETRY_DELAY_SECONDS=2.0",
            "export CAT_AGENT_MAX_FILE_BYTES=65536",
            "export CAT_AGENT_LOG_LEVEL=INFO",
            "export CAT_AGENT_FIREBASE_CREDENTIALS=/opt/firebase/zigbee.json",
            "export CAT_AGENT_FIREBASE_TOKENS=/opt/firebase/tokens.txt",
            "export CAT_AGENT_FIREBASE_TITLE='Гена'",
            "export CAT_AGENT_FIREBASE_TTL=3600",
        )
        for line in expected:
            self.assertIn(line, exported)

    def test_web_settings_are_exported(self) -> None:
        exported = self._capture("web")
        self.assertIn("export CAT_AGENT_WEB_HOST=0.0.0.0", exported)
        self.assertIn("export CAT_AGENT_HTTP_PORT=8080", exported)
        self.assertIn("export CAT_AGENT_WS_PORT=8765", exported)

    def test_voice_settings_are_exported(self) -> None:
        exported = self._capture("voice")
        self.assertIn("CAT_AGENT_WAKE_WORDS_JSON", exported)
        self.assertIn("гена", exported)
        self.assertIn("export CAT_AGENT_NO_COMMAND_TIMEOUT_SECONDS=7.0", exported)
        self.assertIn("export CAT_AGENT_MAX_COMMAND_SECONDS=20.0", exported)

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
