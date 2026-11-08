from __future__ import annotations

from dataclasses import replace
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml

from openai_agent.model_client import OpenAICompatibleChatClient
from openai_agent.runtime import _resolve_api_key
from orchestration.config import Settings
from orchestration.config_env import main as config_env_main
from orchestration.yaml_config import load_app_config
from task_system.readiness import OpenAIReadiness


class ProviderProfilesTest(unittest.TestCase):
    def setUp(self):
        self.fixture = Path(__file__).parent / "fixtures" / "cat-agent.yaml"
        self.source = yaml.safe_load(self.fixture.read_text())

    def _write(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "test.yaml"
        path.write_text(yaml.safe_dump(self.source, allow_unicode=True))
        return path

    def _zai(self):
        self.source["openai"]["active_profile"] = "zai"
        self.source["openai"]["profiles"]["zai"] = {
            "base_url": "https://api.z.ai/api/paas/v4",
            "model": "glm-4.5-flash",
            "readiness": "openai",
            "reasoning_effort": "none",
            "api_key_env": "ZAI_API_KEY",
            "request": {
                "extra_body": {"thinking": {"type": "disabled"}, "max_tokens": 4096},
                "omit_fields": ["top_p"],
                "headers": {"X-Client": "cat-agent"},
            },
        }

    def test_optional_reasoning_effort_and_legacy_profiles(self):
        self.source["openai"]["profiles"]["laptop-12b"]["reasoning_effort"] = "high"
        self.source["openai"]["profiles"]["zai-32b"] = {
            "base_url": "https://api.z.ai/api/paas/v4",
            "model": "glm-4-32b-0414-128k",
            "readiness": "openai_api",
            "api_key_env": "ZAI_API_KEY",
            "request": {"extra_body": {"max_tokens": 4096}},
        }
        cfg = load_app_config(self._write())
        self.assertEqual(cfg.openai.profiles["laptop-12b"].reasoning_effort, "high")
        older = cfg.openai.profiles["zai-32b"]
        self.assertEqual(older.reasoning_effort, "none")
        self.assertEqual(older.request.extra_body, {"max_tokens": 4096})
        self.assertNotIn("thinking", older.request.extra_body)

    def test_existing_profile_keeps_old_defaults(self):
        config = load_app_config(self._write())
        p = config.openai.active
        self.assertIsNone(p.api_key_env)
        self.assertEqual(p.request.extra_body, {})
        self.assertEqual(p.request.omit_fields, ())
        self.assertEqual(p.request.headers, {})

    def test_new_profile_and_json_exports(self):
        self._zai()
        path = self._write()
        p = load_app_config(path).openai.active
        self.assertEqual(p.api_key_env, "ZAI_API_KEY")
        self.assertEqual(p.request.extra_body["thinking"]["type"], "disabled")
        import contextlib
        buf = io.StringIO()
        with patch.dict(os.environ, {"CAT_AGENT_CONFIG": str(path)}):
            with contextlib.redirect_stdout(buf):
                self.assertEqual(config_env_main(["openai"]), 0)
        output = buf.getvalue()
        self.assertIn("CAT_AGENT_OPENAI_API_KEY_ENV=ZAI_API_KEY", output)
        self.assertIn("CAT_AGENT_OPENAI_EXTRA_BODY_JSON=", output)
        self.assertIn("CAT_AGENT_OPENAI_OMIT_FIELDS_JSON=", output)

    def test_invalid_body_and_key_rejected(self):
        self._zai()
        for body in ({"model": "wrong"}, {"messages": []}, {"stream": True}, {"a": float("nan")}):
            with self.subTest(body=body):
                self.source["openai"]["profiles"]["zai"]["request"]["extra_body"] = body
                with self.assertRaises(ValueError):
                    load_app_config(self._write())
        self.source["openai"]["profiles"]["zai"]["request"]["extra_body"] = {}
        self.source["openai"]["profiles"]["zai"]["api_key_env"] = "ZAI-INVALID"
        with self.assertRaises(ValueError):
            load_app_config(self._write())

    def test_request_payload_and_headers_are_generic(self):
        client = OpenAICompatibleChatClient(
            api_base_url="http://127.0.0.1:9000/v1", model="glm-4.5-flash",
            timeout_seconds=4, retries=0, retry_delay_seconds=0,
            max_output_tokens=128000, temperature=0, top_p=1,
            reasoning_effort="none", label="manager", api_key="key",
            extra_body={"thinking": {"type": "disabled"}, "max_tokens": 4096},
            omit_fields=("top_p",), extra_headers={"X-Client": "cat-agent"},
        )
        msg = [{"role": "user", "content": "привет"}]
        payload = client._payload(msg, stream=False)
        self.assertEqual(payload["thinking"], {"type": "disabled"})
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertNotIn("top_p", payload)
        req = client._request(payload)
        self.assertEqual(req.get_header("Authorization"), "Bearer key")
        self.assertEqual(req.get_header("X-client"), "cat-agent")
        self.assertEqual(json.loads(req.data), payload)
        self.assertEqual(client.fork("agent1")._payload(msg, stream=False), payload)

    def test_profile_key_is_strict_and_legacy_fallback_preserved(self):
        settings = Settings.from_env(require_model=False)
        with patch.dict(os.environ, {"ZAI_API_KEY": "zai.secret", "CAT_AGENT_API_KEY": "legacy"}):
            self.assertEqual(_resolve_api_key(replace(settings, api_key_env="ZAI_API_KEY")), "zai.secret")
            self.assertEqual(_resolve_api_key(replace(settings, api_key_env="")), "legacy")
        with patch.dict(os.environ, {"ZAI_API_KEY": "", "CAT_AGENT_API_KEY": "legacy"}):
            with self.assertRaisesRegex(ValueError, "ZAI_API_KEY"):
                _resolve_api_key(replace(settings, api_key_env="ZAI_API_KEY"))

    def test_authenticated_readiness_probes_catalog(self):
        fetch = Mock(return_value={"data": [{"id": "glm-4.5-flash"}]})
        readiness = OpenAIReadiness(
            "https://api.z.ai/api/paas/v4", mode="openai",
            model="glm-4.5-flash", api_key="some-key", fetch=fetch,
        )
        self.assertTrue(readiness.check().ready)
        fetch.assert_called_once_with(
            "https://api.z.ai/api/paas/v4/models",
            headers={"Authorization": "Bearer some-key"}, timeout=3.0,
        )


if __name__ == "__main__":
    unittest.main()
