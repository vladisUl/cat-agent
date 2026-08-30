from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import time
from typing import Any, Callable
from urllib import error, request

from orchestration.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    phase: str
    phase_started: float | None
    prefill_seconds: float | None
    generation_seconds: float | None
    total_seconds: float | None
    finished_at: float | None


ModelEventHandler = Callable[[str, str, str], None]


class OpenAICompatibleChatClient:
    """OpenAI-compatible /v1/chat/completions adapter for cat-agent CORE."""

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
        *,
        label: str,
        api_key: str = "",
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
        self.label = label
        self.api_key = api_key.strip()
        self.stream_enabled = os.getenv(
            "CAT_AGENT_OPENAI_STREAM", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.max_tokens_field = (
            os.getenv("CAT_AGENT_OPENAI_MAX_TOKENS_FIELD", "max_tokens").strip()
            or "max_tokens"
        )

        self._resident_tokens = 0
        self._last_response: ChatResponse | None = None
        self._event_handler: ModelEventHandler | None = None
        self._inference_timing = InferenceTiming("idle", None, None, None, None, None)

    @property
    def chat_completions_url(self) -> str:
        return f"{self.api_base_url}/chat/completions"

    @property
    def resident_tokens(self) -> int:
        return self._resident_tokens

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    @property
    def inference_timing(self) -> InferenceTiming:
        return self._inference_timing

    def set_event_handler(self, handler: ModelEventHandler | None) -> None:
        self._event_handler = handler

    def wait_until_ready(
        self,
        stop_requested: Callable[[], bool],
        interval: float = 2.0,
    ) -> bool:
        del interval
        return not stop_requested()

    def close(self) -> None:
        self._resident_tokens = 0
        self._last_response = None
        self._event_handler = None
        self._inference_timing = InferenceTiming("idle", None, None, None, None, None)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        del messages
        # A remote OpenAI-compatible endpoint owns no cat-agent resident KV slot.
        self._resident_tokens = 0
        self._last_response = None

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        if not messages or messages[-1].get("role") != "user":
            raise ModelClientError("OpenAI-compatible turn must end in a user message")
        stream = self._event_handler is not None and self.stream_enabled
        return self._complete(messages, stream=stream)

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
    ) -> ChatResponse:
        started = time.monotonic()
        self._inference_timing = InferenceTiming(
            "prefill", started, None, None, None, None
        )
        self._emit_event("prefill_start")

        try:
            if stream:
                content, usage, first_chunk_at = self._stream_request(messages)
            else:
                response = self._post_json(self._payload(messages, stream=False))
                content = _response_content(response)
                usage = _usage(response)
                first_chunk_at = None
                self._emit_event("chunk", content)

            finished = time.monotonic()
            elapsed = finished - started
            if first_chunk_at is None:
                prefill_seconds = None
                generation_seconds = elapsed
            else:
                prefill_seconds = first_chunk_at - started
                generation_seconds = max(finished - first_chunk_at, 0.0)

            prompt_tokens = _int_or_none(usage.get("prompt_tokens"))
            completion_tokens = _int_or_none(usage.get("completion_tokens"))
            cached_tokens = _cached_tokens(usage)
            prompt_evaluated_tokens: int | None = None
            if prompt_tokens is not None and cached_tokens is not None:
                prompt_evaluated_tokens = max(prompt_tokens - cached_tokens, 0)

            # Remote providers do not expose a cat-agent-owned resident KV slot.
            self._resident_tokens = 0

            self._inference_timing = InferenceTiming(
                "idle",
                None,
                prefill_seconds,
                generation_seconds,
                elapsed,
                finished,
            )
            self._emit_event("decode_done")

            result = ChatResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed,
                cached_tokens=cached_tokens,
                prompt_evaluated_tokens=prompt_evaluated_tokens,
                prompt_seconds=prefill_seconds,
                generation_seconds=generation_seconds,
            )
            self._last_response = result
            return result
        except ModelClientError:
            self._finish_failed_timing(started)
            raise
        except Exception as exc:
            self._finish_failed_timing(started)
            raise ModelClientError(f"OpenAI-compatible request failed: {exc}") from exc

    def _payload(
        self,
        messages: list[dict[str, str]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            self.max_tokens_field: self.max_output_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.reasoning_effort != "none":
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _stream_request(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any], float | None]:
        payload = self._payload(messages, stream=True)
        http_request = self._request(payload)
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        first_chunk_at: float | None = None

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        item = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise ModelClientError(
                            f"Invalid OpenAI-compatible stream JSON: {data!r}"
                        ) from exc
                    if not isinstance(item, dict):
                        continue

                    event_usage = _usage(item)
                    if event_usage:
                        usage = event_usage

                    chunk = _stream_content(item)
                    if not chunk:
                        continue
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                        self._inference_timing = InferenceTiming(
                            "generate", first_chunk_at, None, None, None, None
                        )
                        self._emit_event("decode_start")
                    chunks.append(chunk)
                    self._emit_event("chunk", chunk)
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(
                f"HTTP {exc.code} from OpenAI-compatible endpoint: {body}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise ModelClientError(f"OpenAI-compatible stream failed: {exc}") from exc

        return "".join(chunks), usage, first_chunk_at

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._send_json(self._request(payload))

    def _request(self, payload: dict[str, Any]) -> request.Request:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return request.Request(
            self.chat_completions_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=headers,
        )

    def _send_json(self, http_request: request.Request) -> dict[str, Any]:
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
                message = (
                    f"HTTP {exc.code} from OpenAI-compatible endpoint: {body}"
                )
                if 400 <= exc.code < 500:
                    raise ModelClientError(message) from exc
                last_error = ModelClientError(message)
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < attempts:
                delay = self.retry_delay_seconds * attempt
                LOGGER.warning(
                    "OpenAI-compatible request failed (%s/%s): %s; retry in %.1f s",
                    attempt,
                    attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise ModelClientError(f"OpenAI-compatible request failed: {last_error}")

    def _emit_event(self, event: str, payload: str = "") -> None:
        handler = self._event_handler
        if handler is None:
            return
        try:
            handler(self.label, event, payload)
        except Exception:
            LOGGER.exception("OpenAI-compatible model event handler failed")

    def _finish_failed_timing(self, started: float) -> None:
        finished = time.monotonic()
        self._inference_timing = InferenceTiming(
            "idle",
            None,
            None,
            None,
            finished - started,
            finished,
        )


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelClientError(
            f"Unexpected OpenAI-compatible response structure: {response!r}"
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ModelClientError(
            f"Unexpected OpenAI-compatible choice structure: {choice!r}"
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ModelClientError(
            f"Unexpected OpenAI-compatible message structure: {choice!r}"
        )
    return _content_text(message.get("content"))


def _stream_content(item: dict[str, Any]) -> str:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return ""
    return _content_text(delta.get("content"))


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def _cached_tokens(usage: dict[str, Any]) -> int | None:
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    return _int_or_none(details.get("cached_tokens"))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None
