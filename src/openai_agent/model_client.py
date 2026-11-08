from __future__ import annotations

from dataclasses import dataclass
import json
from agent_core.metrics import measured
import logging
import os
import time
from typing import Any, Callable
from uuid import uuid4
from urllib import error, request

from orchestration.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


from agent_core.types import InferenceTiming


ModelEventHandler = Callable[[str, str, str], None]


class OpenAICompatibleChatClient:
    supports_images = True
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
        extra_body: dict[str, Any] | None = None,
        omit_fields: tuple[str, ...] = (),
        extra_headers: dict[str, str] | None = None,
        trace_enabled: bool | None = None,
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
        self.extra_body = dict(extra_body or {})
        self.omit_fields = tuple(omit_fields)
        self.extra_headers = dict(extra_headers or {})
        self.trace_enabled = (
            os.getenv("CAT_AGENT_OPENAI_TRACE", "0").strip().lower()
            in {"1", "true", "yes", "on"}
            if trace_enabled is None else trace_enabled
        )
        protected = {"model", "messages", "stream"}
        if protected.intersection(self.extra_body) or protected.intersection(self.omit_fields):
            raise ValueError("request options must not override model, messages or stream")
        if any(key.lower() in {"authorization", "content-type", "host"} for key in self.extra_headers):
            raise ValueError("extra headers must not override Authorization, Content-Type or Host")
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

    def fork(self, label, *, inherit_base: bool = True):
        del inherit_base
        return OpenAICompatibleChatClient(
            api_base_url=self.api_base_url, model=self.model,
            timeout_seconds=self.timeout_seconds, retries=self.retries,
            retry_delay_seconds=self.retry_delay_seconds,
            max_output_tokens=self.max_output_tokens, temperature=self.temperature,
            top_p=self.top_p, reasoning_effort=self.reasoning_effort,
            label=label, api_key=self.api_key,
            extra_body=self.extra_body, omit_fields=self.omit_fields,
            extra_headers=self.extra_headers,
            trace_enabled=self.trace_enabled,
        )

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

    @measured("model_seconds")
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
        payload.update(self.extra_body)
        for field in self.omit_fields:
            payload.pop(field, None)
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
        trace_id = uuid4().hex[:12] if self.trace_enabled else ""
        trace_lines: list[str] = []
        http_status: int | None = None
        if self.trace_enabled:
            self._trace_request(trace_id, http_request, attempt=1)
        http_started = time.monotonic()

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                http_status = response.status
                for raw in response:
                    if self.trace_enabled:
                        trace_lines.append(raw.decode("utf-8", errors="replace"))
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
            http_status = exc.code
            body = exc.read().decode("utf-8", errors="replace")
            if self.trace_enabled:
                trace_lines.append(body)
            raise ModelClientError(
                f"HTTP {exc.code} from OpenAI-compatible endpoint: {body}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise ModelClientError(f"OpenAI-compatible stream failed: {exc}") from exc
        finally:
            if self.trace_enabled:
                self._trace_response(
                    trace_id, http_status, time.monotonic() - http_started,
                    "".join(trace_lines), stream=True,
                )

        return "".join(chunks), usage, first_chunk_at

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._send_json(self._request(payload))

    def _request(self, payload: dict[str, Any]) -> request.Request:
        headers = {"Content-Type": "application/json", **self.extra_headers}
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
        trace_id = uuid4().hex[:12] if self.trace_enabled else ""

        for attempt in range(1, attempts + 1):
            if self.trace_enabled:
                self._trace_request(trace_id, http_request, attempt=attempt)
            http_started = time.monotonic()
            try:
                with request.urlopen(
                    http_request,
                    timeout=self.timeout_seconds,
                ) as response:
                    body = response.read().decode("utf-8")
                    if self.trace_enabled:
                        self._trace_response(
                            trace_id, response.status, time.monotonic() - http_started,
                            body, stream=False,
                        )
                    decoded = json.loads(body)
                    if not isinstance(decoded, dict):
                        raise ModelClientError(
                            f"Expected JSON object, got {type(decoded).__name__}"
                        )
                    return decoded
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if self.trace_enabled:
                    self._trace_response(
                        trace_id, exc.code, time.monotonic() - http_started,
                        body, stream=False,
                    )
                message = (
                    f"HTTP {exc.code} from OpenAI-compatible endpoint: {body}"
                )
                if 400 <= exc.code < 500:
                    raise ModelClientError(message) from exc
                last_error = ModelClientError(message)
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if self.trace_enabled:
                    LOGGER.info(
                        "OPENAI TRACE id=%s role=%s transport_or_decode_error seconds=%.3f type=%s",
                        trace_id, self.label, time.monotonic() - http_started,
                        type(exc).__name__,
                    )
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

    def _trace_request(self, trace_id: str, http_request: request.Request, *, attempt: int) -> None:
        # Only the serialized POST body is logged: never HTTP authentication headers.
        LOGGER.info(
            "OPENAI TRACE id=%s role=%s attempt=%d REQUEST url=%s body=%s",
            trace_id, self.label, attempt, http_request.full_url,
            http_request.data.decode("utf-8"),
        )

    def _trace_response(
        self, trace_id: str, status: int | None, elapsed: float,
        body: str, *, stream: bool,
    ) -> None:
        LOGGER.info(
            "OPENAI TRACE id=%s role=%s RESPONSE status=%s seconds=%.3f stream=%s body=%s",
            trace_id, self.label, status, elapsed, stream, body,
        )
        # This is diagnostic metadata, never substituted for the raw response.
        # Absence of reasoning_content is NOT proof of zero internal reasoning.
        events: list[dict[str, Any]] = []
        if stream:
            for line in body.splitlines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw or raw == "[DONE]":
                    continue
                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    events.append(item)
        else:
            try:
                item = json.loads(body)
            except json.JSONDecodeError:
                item = None
            if isinstance(item, dict):
                events.append(item)

        usage: dict[str, Any] = {}
        reasoning_chars = 0
        reasoning_present = False
        for item in events:
            if isinstance(item.get("usage"), dict):
                usage = item["usage"]
            choices = item.get("choices")
            if not isinstance(choices, list):
                continue
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                part = choice.get("delta" if stream else "message")
                if not isinstance(part, dict) or "reasoning_content" not in part:
                    continue
                reasoning_present = True
                value = part["reasoning_content"]
                if isinstance(value, str):
                    reasoning_chars += len(value)
        LOGGER.info(
            "OPENAI TRACE id=%s role=%s SUMMARY usage=%s reasoning_content_present=%s reasoning_content_chars=%d",
            trace_id, self.label, json.dumps(usage, ensure_ascii=False),
            reasoning_present, reasoning_chars,
        )

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

