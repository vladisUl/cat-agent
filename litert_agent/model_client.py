from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable
from urllib import error, request

from cat_agent.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


class LiteRTChatClient:
    """OpenAI-compatible LiteRT-LM server client.

    This first correctness pass sends the full message history on every request.
    Native LiteRT-LM Conversation/KV reuse is investigated separately after the
    complete manager -> agent -> tool -> agent -> manager chain works.
    """

    def __init__(
        self,
        api_base_url: str,
        model: str,
        timeout_seconds: int,
        retries: int,
        retry_delay_seconds: float,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
    ) -> None:
        base = api_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        self.api_base_url = base
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort

    @property
    def models_url(self) -> str:
        return f"{self.api_base_url}/models"

    @property
    def chat_url(self) -> str:
        return f"{self.api_base_url}/chat/completions"

    def wait_until_ready(
        self, stop_requested: Callable[[], bool], interval: float = 2.0
    ) -> bool:
        while not stop_requested():
            try:
                http_request = request.Request(self.models_url, method="GET")
                response = self._send(http_request)
                if not isinstance(response.get("data"), list):
                    raise ModelClientError(
                        f"Unexpected /v1/models response from LiteRT-LM: {response!r}"
                    )
                LOGGER.info("LiteRT-LM server is ready")
                return True
            except ModelClientError as exc:
                LOGGER.info("Waiting for LiteRT-LM server: %s", exc)
                time.sleep(interval)
        return False

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        started = time.monotonic()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_output_tokens,
        }
        response = self._post_json(self.chat_url, payload)
        elapsed = time.monotonic() - started

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelClientError(
                f"Unexpected LiteRT-LM chat response structure: {response!r}"
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise ModelClientError(
                f"Unexpected LiteRT-LM choice structure: {response!r}"
            )
        message = first.get("message")
        if not isinstance(message, dict):
            raise ModelClientError(
                f"Unexpected LiteRT-LM message structure: {response!r}"
            )
        content = message.get("content")
        if not isinstance(content, str):
            raise ModelClientError(
                f"Unexpected LiteRT-LM content structure: {response!r}"
            )

        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        usage = response.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
            completion_tokens = _int_or_none(usage.get("completion_tokens"))

        # Stock OpenAI-compatible responses do not split server time into
        # prefill/decode and do not report reused KV tokens. Do not fabricate
        # those metrics; phase 2 will instrument the native Conversation path.
        return ChatResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_seconds=elapsed,
            cached_tokens=None,
            prompt_evaluated_tokens=None,
            prompt_seconds=None,
            generation_seconds=None,
        )

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=encoded,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._send(http_request)

    def _send(self, http_request: request.Request) -> dict[str, Any]:
        last_error: Exception | None = None
        attempts = self.retries + 1

        for attempt in range(1, attempts + 1):
            try:
                with request.urlopen(
                    http_request,
                    timeout=self.timeout_seconds,
                ) as response:
                    body = response.read().decode("utf-8")
                    decoded = json.loads(body)
                    if not isinstance(decoded, dict):
                        raise ModelClientError(
                            f"Expected JSON object, got {type(decoded).__name__}"
                        )
                    return decoded
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                message = f"HTTP {exc.code} from LiteRT-LM server: {body}"
                if 400 <= exc.code < 500:
                    raise ModelClientError(message) from exc
                last_error = ModelClientError(message)
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < attempts:
                delay = self.retry_delay_seconds * attempt
                LOGGER.warning(
                    "LiteRT-LM request failed (%s/%s): %s; retry in %.1f s",
                    attempt,
                    attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise ModelClientError(f"LiteRT-LM request failed: {last_error}")


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
