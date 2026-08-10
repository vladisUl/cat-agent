from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Callable
from urllib import error, request

LOGGER = logging.getLogger(__name__)


class ModelClientError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ChatResponse:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    elapsed_seconds: float


class OpenAIChatClient:
    """Small OpenAI-compatible text chat client.

    It deliberately does not send tools/tool_choice and does not consume
    tool_calls. The model server is used only as a text-generation backend.

    When ``id_slot`` is set, llama-server requests are pinned to that slot and
    prompt caching is enabled for the slot. This keeps independent long-lived
    KV caches for the manager and agent roles.
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
        id_slot: int | None = None,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.id_slot = id_slot

    @property
    def chat_url(self) -> str:
        return f"{self.api_base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.api_base_url}/models"

    def list_models(self) -> dict[str, Any]:
        http_request = request.Request(self.models_url, method="GET")
        return self._send(http_request)

    def wait_until_ready(
        self, stop_requested: Callable[[], bool], interval: float = 2.0
    ) -> bool:
        while not stop_requested():
            try:
                response = self.list_models()
                model_ids = {
                    item.get("id")
                    for item in response.get("data", [])
                    if isinstance(item, dict)
                }
                if model_ids and self.model not in model_ids:
                    LOGGER.warning(
                        "Model server is ready, but model %r is not listed. "
                        "Available models: %s",
                        self.model,
                        sorted(model_ids),
                    )
                else:
                    LOGGER.info("Model server is ready")
                return True
            except ModelClientError as exc:
                LOGGER.info("Waiting for model server: %s", exc)
                time.sleep(interval)
        return False

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.id_slot is not None:
            payload["id_slot"] = self.id_slot
            payload["cache_prompt"] = True

        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.chat_url,
            data=encoded,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        started = time.monotonic()
        response = self._send(http_request)
        elapsed = time.monotonic() - started

        try:
            message = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelClientError(
                f"Unexpected chat response structure: {response!r}"
            ) from exc

        if not isinstance(message, dict):
            raise ModelClientError(f"Response message is not an object: {message!r}")

        content = message.get("content")
        if not isinstance(content, str):
            raise ModelClientError(
                "Model returned no text content. Native tool calls are not used "
                f"by this agent: {message!r}"
            )

        usage = response.get("usage")
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        if isinstance(usage, dict):
            prompt_value = usage.get("prompt_tokens")
            completion_value = usage.get("completion_tokens")
            if isinstance(prompt_value, int):
                prompt_tokens = prompt_value
            if isinstance(completion_value, int):
                completion_tokens = completion_value

        return ChatResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_seconds=elapsed,
        )

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
                message = f"HTTP {exc.code} from model server: {body}"
                if 400 <= exc.code < 500:
                    raise ModelClientError(message) from exc
                last_error = ModelClientError(message)
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < attempts:
                delay = self.retry_delay_seconds * attempt
                LOGGER.warning(
                    "Model request failed (%s/%s): %s; retry in %.1f s",
                    attempt,
                    attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise ModelClientError(f"Model request failed: {last_error}")
