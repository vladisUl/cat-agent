"""CORE health reporting; independent of model execution and user interfaces."""
import threading
import time

from .ollama_usage import CheckResult, OllamaUsageProbe, fetch_json, HTTP_TIMEOUT

HEARTBEAT_SECONDS = 5.0
PROBE_SECONDS = 30.0
REASONS = {'', 'initializing', 'stopping', 'runtime_unavailable', 'readiness_unknown',
           'missing_api_key', 'credentials_invalid', 'quota_exhausted',
           'usage_api_error', 'usage_invalid_response', 'usage_network_error',
           'ollama_unavailable', 'ollama_invalid_response',
           'openai_unavailable', 'openai_invalid_response', 'openai_model_unavailable'}


class OpenAIReadiness:
    def __init__(self, api_base_url, *, mode='ollama', model='', fetch=fetch_json, usage=None):
        base = api_base_url.rstrip('/')
        self.mode = mode
        self.model = model
        self.fetch = fetch
        self.usage = usage or OllamaUsageProbe()
        if mode == 'ollama':
            self.url = (base[:-3] if base.endswith('/v1') else base) + '/api/tags'
        elif mode == 'openai':
            self.url = base + '/models'
        else:
            raise ValueError(f'unsupported OpenAI readiness mode: {mode!r}')

    def check(self):
        if self.mode == 'ollama':
            try:
                data = self.fetch(self.url, timeout=HTTP_TIMEOUT)
            except Exception:
                return CheckResult(False, 'ollama_unavailable')
            if not isinstance(data, dict) or not isinstance(data.get('models'), list):
                return CheckResult(False, 'ollama_invalid_response')
            return self.usage.check()

        try:
            data = self.fetch(self.url, timeout=HTTP_TIMEOUT)
        except Exception:
            return CheckResult(False, 'openai_unavailable')
        models = data.get('data') if isinstance(data, dict) else None
        if not isinstance(models, list):
            return CheckResult(False, 'openai_invalid_response')
        ids = {
            item.get('id')
            for item in models
            if isinstance(item, dict) and isinstance(item.get('id'), str)
        }
        if self.model and self.model not in ids:
            return CheckResult(False, 'openai_model_unavailable')
        return CheckResult(True)


class HealthReporter:
    def __init__(self, publish, runtime_alive, *, probe=None, clock=time.monotonic):
        self.publish = publish
        self.runtime_alive = runtime_alive
        self.probe = probe
        self.clock = clock
        self.result = CheckResult(False, 'initializing')
        self.next_probe = 0.0
        self.stop = threading.Event()
        self.thread = None

    def report_once(self):
        if not self.runtime_alive():
            self.result = CheckResult(False, 'runtime_unavailable')
            self.next_probe = 0.0
        elif self.probe is None:
            self.result = CheckResult(True)
        elif self.clock() >= self.next_probe:
            # Publish the new evidence only when the entire check completes.
            self.result = self.probe.check()
            self.next_probe = self.clock() + PROBE_SECONDS
        if not self.stop.is_set():
            self.publish(self.result)

    def _run(self):
        while not self.stop.is_set():
            try:
                self.report_once()
            except Exception:
                # A reporter/API failure cannot leak credential-bearing exceptions
                # or preserve a READY lease indefinitely: no heartbeat => expiry.
                self.result = CheckResult(False, 'readiness_unknown')
                self.next_probe = 0.0
            self.stop.wait(HEARTBEAT_SECONDS)

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name='core-readiness', daemon=True)
            self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=10)
