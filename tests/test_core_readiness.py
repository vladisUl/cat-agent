import io
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from task_system.ollama_usage import OllamaUsageProbe, CheckResult, NoRedirect, USAGE_URL
from task_system.readiness import OpenAIReadiness, HealthReporter


def usage_payload(value):
    return {'limits': {'monthly': {'usage': value, 'models': [{'name': 'gemma4:31b', 'request_count': 59}]}}}


class OllamaReadinessTest(unittest.TestCase):
    def probe(self, response=None, error=None):
        fetch = Mock(return_value=response, side_effect=error)
        return OllamaUsageProbe(fetch), fetch

    def check(self, probe):
        with patch.dict('os.environ', {'OLLAMA_API_KEY': 'test-secret-key'}):
            return probe.check()

    def test_usage_0008_is_ready_and_request_is_authenticated(self):
        probe, fetch = self.probe(usage_payload(.008))
        self.assertEqual(self.check(probe), CheckResult(True))
        fetch.assert_called_once_with(USAGE_URL, headers={'Authorization': 'Bearer test-secret-key'}, timeout=3.0)

    def test_usage_one_or_more_is_exhausted(self):
        for value in (1.0, 1.2, 10):
            with self.subTest(value=value):
                probe, _ = self.probe(usage_payload(value))
                self.assertEqual(self.check(probe), CheckResult(False, 'quota_exhausted'))

    def test_401_403_are_invalid_credentials(self):
        for status in (401, 403):
            with self.subTest(status=status):
                probe, _ = self.probe(error=HTTPError(USAGE_URL, status, 'error', {}, None))
                self.assertEqual(self.check(probe), CheckResult(False, 'credentials_invalid'))

    def test_http_api_errors_are_not_ready(self):
        for status in (404, 429, 500, 503):
            with self.subTest(status=status):
                probe, _ = self.probe(error=HTTPError(USAGE_URL, status, 'error', {}, None))
                self.assertEqual(self.check(probe), CheckResult(False, 'usage_api_error'))

    def test_network_and_timeout_errors_are_not_ready(self):
        for error in (URLError('unreachable'), TimeoutError('timeout'), OSError('network error')):
            with self.subTest(error=type(error).__name__):
                probe, _ = self.probe(error=error)
                self.assertEqual(self.check(probe), CheckResult(False, 'usage_network_error'))

    def test_invalid_or_changed_usage_schema_is_not_ready(self):
        cases = [{}, None, [], {'limits': {'monthly': {}}}]
        cases += [usage_payload(value) for value in ('0.008', None, True, float('nan'), float('inf'), -1)]
        for data in cases:
            with self.subTest(data=data):
                probe, _ = self.probe(data)
                self.assertEqual(self.check(probe), CheckResult(False, 'usage_invalid_response'))
        probe, _ = self.probe(error=ValueError('invalid JSON'))
        self.assertFalse(self.check(probe).ready)

    def test_missing_key_makes_no_usage_request(self):
        probe, fetch = self.probe(usage_payload(.008))
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(probe.check(), CheckResult(False, 'missing_api_key'))
        fetch.assert_not_called()

    def test_key_or_error_payload_never_reaches_diagnostics(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            probe, _ = self.probe(error=URLError('sensitive test-secret-key response'))
            result = self.check(probe)
            self.assertNotIn('test-secret-key', repr(result))
            self.assertNotIn('test-secret-key', stream.getvalue())
        finally:
            logger.removeHandler(handler)

    def test_usage_authentication_is_not_forwarded_on_redirect(self):
        request = SimpleNamespace(full_url=USAGE_URL)
        with self.assertRaises(HTTPError):
            NoRedirect().redirect_request(request, None, 302, 'redirect', {}, 'https://other.example/')

    def test_local_ollama_and_usage_must_both_succeed(self):
        quota = Mock()
        quota.check.return_value = CheckResult(True)
        fetch = Mock(return_value={'models': []})
        probe = OpenAIReadiness('http://127.0.0.1:11434/v1/', fetch=fetch, usage=quota)
        self.assertTrue(probe.check().ready)
        fetch.assert_called_once_with('http://127.0.0.1:11434/api/tags', timeout=3.0)
        quota.check.assert_called_once_with()
        quota.check.return_value = CheckResult(False, 'quota_exhausted')
        self.assertEqual(probe.check().reason, 'quota_exhausted')

    def test_local_ollama_unavailable_even_if_quota_ok(self):
        quota = Mock()
        probe = OpenAIReadiness('http://localhost:11434/v1', fetch=Mock(side_effect=TimeoutError()), usage=quota)
        self.assertEqual(probe.check(), CheckResult(False, 'ollama_unavailable'))
        quota.check.assert_not_called()
        probe.fetch = Mock(return_value={'unexpected': True})
        self.assertEqual(probe.check().reason, 'ollama_invalid_response')


class HealthReporterTest(unittest.TestCase):
    def test_litert_ready_only_after_runtime_initialization(self):
        alive = False
        reports = []
        reporter = HealthReporter(reports.append, lambda: alive)
        reporter.report_once()
        self.assertFalse(reports[-1].ready)
        alive = True
        reporter.report_once()
        self.assertTrue(reports[-1].ready)
        alive = False
        reporter.report_once()
        self.assertFalse(reports[-1].ready)

    def test_openai_probe_periodic_and_failure_replaces_cached_ready(self):
        now = 0
        reports = []
        probe = Mock()
        probe.check.return_value = CheckResult(True)
        reporter = HealthReporter(reports.append, lambda: True, probe=probe, clock=lambda: now)
        reporter.report_once()
        now = 5
        reporter.report_once()
        self.assertEqual(probe.check.call_count, 1)
        probe.check.return_value = CheckResult(False, 'usage_network_error')
        now = 30
        reporter.report_once()
        self.assertEqual(probe.check.call_count, 2)
        self.assertEqual(reports[-1], CheckResult(False, 'usage_network_error'))

    def test_stopped_reporter_cannot_resurrect_ready(self):
        reports = []
        reporter = HealthReporter(reports.append, lambda: True)
        reporter.close()
        reporter.report_once()
        self.assertEqual(reports, [])

    def test_failed_core_warmup_never_starts_heartbeat(self):
        from agent_core.core_scheduler import CoreScheduler
        system = SimpleNamespace(is_remote=True, start_worker=Mock())
        scheduler = CoreScheduler(SimpleNamespace(system_runtime=system, manager_client=Mock()))
        self.addCleanup(scheduler._executor.shutdown, wait=True)
        scheduler._prepare_contexts = Mock(side_effect=RuntimeError('model initialization failed'))
        with self.assertRaises(RuntimeError):
            scheduler.start()
        system.start_worker.assert_not_called()
