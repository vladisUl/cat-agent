"""2026 Ollama Cloud session/weekly readiness schema and legacy monthly support."""
from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from task_system.ollama_usage import CheckResult, OllamaUsageProbe
from task_system.readiness import OpenAIReadiness


class OllamaCloudUsageSchemaTests(unittest.TestCase):
    def check(self, limits):
        fetch = Mock(return_value={"limits": limits})
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "unit-test-only"}):
            return OllamaUsageProbe(fetch).check()

    def test_new_session_and_weekly_windows_ready(self):
        self.assertEqual(
            self.check({
                "session": {"usage": 0.24, "models": []},
                "weekly": {"usage": 0.53, "models": []},
            }),
            CheckResult(True),
        )

    def test_session_limit_exhausted(self):
        self.assertEqual(
            self.check({"session": {"usage": 1.0}, "weekly": {"usage": 0.2}}),
            CheckResult(False, "quota_exhausted"),
        )

    def test_weekly_limit_exhausted(self):
        self.assertEqual(
            self.check({"session": {"usage": 0.1}, "weekly": {"usage": 1.0}}),
            CheckResult(False, "quota_exhausted"),
        )

    def test_monthly_legacy_format_remains_ready(self):
        self.assertEqual(self.check({"monthly": {"usage": 0.2}}), CheckResult(True))
        self.assertEqual(
            self.check({"monthly": {"usage": 1.0}}),
            CheckResult(False, "quota_exhausted"),
        )

    def test_combined_schema_checks_all_limits(self):
        self.assertEqual(
            self.check({
                "monthly": {"usage": 0.1},
                "session": {"usage": 1.0},
                "weekly": {"usage": 0.1},
            }),
            CheckResult(False, "quota_exhausted"),
        )

    def test_partial_window_and_malformed_usage_never_claim_ready(self):
        for limits in (
            {}, [], None,
            {"session": {"usage": 0.1}},
            {"weekly": {"usage": 0.1}},
            {"monthly": {"usage": 0.1}, "session": {"usage": 0.1}},
            {"session": {}, "weekly": {"usage": 0.1}},
            {"session": {"usage": 0.1}, "weekly": {}},
            {"session": {"usage": True}, "weekly": {"usage": 0.1}},
            {"session": {"usage": 0.1}, "weekly": {"usage": "0.3"}},
            {"session": {"usage": float("nan")}, "weekly": {"usage": 0.1}},
            {"session": {"usage": 0.1}, "weekly": {"usage": -0.1}},
        ):
            with self.subTest(limits=limits):
                self.assertEqual(
                    self.check(limits),
                    CheckResult(False, "usage_invalid_response"),
                )

    def test_local_ollama_ready_uses_new_cloud_usage_evidence(self):
        usage = OllamaUsageProbe(Mock(return_value={
            "limits": {"session": {"usage": 0.3}, "weekly": {"usage": 0.5}}
        }))
        readiness = OpenAIReadiness(
            "http://127.0.0.1:11434/v1", mode="ollama",
            fetch=Mock(return_value={"models": []}), usage=usage,
        )
        with patch.dict(os.environ, {"OLLAMA_API_KEY": "unit-test-only"}):
            self.assertEqual(readiness.check(), CheckResult(True))


if __name__ == "__main__":
    unittest.main()
