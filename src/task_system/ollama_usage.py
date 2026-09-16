"""Replaceable adapter for Ollama's undocumented account usage endpoint.

Never return response bodies, request headers, keys or exception text to logs/RPC.
"""
from dataclasses import dataclass
import json
import math
import os
from urllib.error import HTTPError
from urllib.request import Request, HTTPRedirectHandler, build_opener

USAGE_URL = 'https://ollama.com/api/usage'
HTTP_TIMEOUT = 3.0


@dataclass(frozen=True)
class CheckResult:
    ready: bool
    reason: str = ''


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # In particular, never forward the account key to a redirected host.
        raise HTTPError(req.full_url, code, 'redirect refused', headers, fp)


def fetch_json(url, *, headers=None, timeout=HTTP_TIMEOUT):
    request = Request(url, headers=headers or {}, method='GET')
    with build_opener(NoRedirect()).open(request, timeout=timeout) as response:
        raw = response.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError('response too large')
        return json.loads(raw)


class OllamaUsageProbe:
    def __init__(self, fetch=fetch_json):
        self.fetch = fetch

    def check(self):
        key = os.environ.get('OLLAMA_API_KEY', '').strip()
        if not key:
            return CheckResult(False, 'missing_api_key')
        try:
            data = self.fetch(USAGE_URL, headers={'Authorization': 'Bearer ' + key}, timeout=HTTP_TIMEOUT)
        except HTTPError as exc:
            return CheckResult(False, 'credentials_invalid' if exc.code in (401, 403) else 'usage_api_error')
        except (ValueError, TypeError):
            return CheckResult(False, 'usage_invalid_response')
        except Exception:
            return CheckResult(False, 'usage_network_error')
        try:
            usage = data['limits']['monthly']['usage']
            if isinstance(usage, bool) or not isinstance(usage, (int, float)) or not math.isfinite(usage) or usage < 0:
                raise ValueError('invalid usage')
        except (KeyError, TypeError, ValueError, OverflowError):
            return CheckResult(False, 'usage_invalid_response')
        return CheckResult(False, 'quota_exhausted') if usage >= 1.0 else CheckResult(True)
