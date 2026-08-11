from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
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
    cached_tokens: int | None = None
    prompt_evaluated_tokens: int | None = None
    prompt_seconds: float | None = None
    generation_seconds: float | None = None


class OpenAIChatClient:
    """Native llama-server text chat client.

    The class name is intentionally kept for this experiment so the rest of the
    runtime does not change. Messages are formatted by llama-server itself via
    /apply-template, then sent directly to the native /completion endpoint.

    When ``id_slot`` is set, requests are pinned to that llama-server slot and
    prompt caching is enabled. This keeps independent long-lived KV caches for
    the manager and agent roles.
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
        base = api_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self.api_base_url = base.rstrip("/")
        # Kept in the constructor for compatibility with the existing runtime.
        # Native llama-server already has the model loaded and reasoning is
        # disabled globally by start_server.sh.
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.id_slot = id_slot

        # Experimental diagnostics only. When enabled, tokenize each templated
        # prompt and remember the token sequence that should currently reside in
        # this client's fixed slot (prompt + generated tokens). This lets us
        # compare the theoretical common prefix with llama-server's actual
        # prompt work on the next request without changing agent behaviour.
        self.trace_prompt_reuse = os.getenv(
            "CAT_AGENT_TRACE_PROMPT_REUSE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._slot_tokens: list[int] | None = None

    @property
    def health_url(self) -> str:
        return f"{self.api_base_url}/health"

    @property
    def apply_template_url(self) -> str:
        return f"{self.api_base_url}/apply-template"

    @property
    def completion_url(self) -> str:
        return f"{self.api_base_url}/completion"

    @property
    def tokenize_url(self) -> str:
        return f"{self.api_base_url}/tokenize"

    def wait_until_ready(
        self, stop_requested: Callable[[], bool], interval: float = 2.0
    ) -> bool:
        while not stop_requested():
            try:
                http_request = request.Request(self.health_url, method="GET")
                response = self._send(http_request)
                if response.get("status") != "ok":
                    raise ModelClientError(
                        f"Unexpected health response from model server: {response!r}"
                    )
                LOGGER.info("Model server is ready")
                return True
            except ModelClientError as exc:
                LOGGER.info("Waiting for model server: %s", exc)
                time.sleep(interval)
        return False

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        return self._complete(messages, n_predict=self.max_output_tokens)

    def prefill(self, messages: list[dict[str, str]]) -> ChatResponse:
        """Evaluate a templated conversation into the assigned slot without decoding."""
        return self._complete(messages, n_predict=0)

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        n_predict: int,
    ) -> ChatResponse:
        started = time.monotonic()
        prompt = self._apply_template(messages)

        prompt_token_ids: list[int] | None = None
        expected_common: int | None = None
        resident_before: int | None = None
        if self.trace_prompt_reuse:
            prompt_token_ids = self._tokenize(prompt)
            if self._slot_tokens is not None:
                resident_before = len(self._slot_tokens)
                expected_common = _common_prefix_length(
                    self._slot_tokens,
                    prompt_token_ids,
                )

        payload: dict[str, Any] = {
            "prompt": prompt,
            "stream": False,
            "n_predict": n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cache_prompt": True,
        }
        if self.trace_prompt_reuse:
            payload["return_tokens"] = True
        if self.id_slot is not None:
            payload["id_slot"] = self.id_slot

        response = self._post_json(self.completion_url, payload)
        elapsed = time.monotonic() - started

        content = response.get("content")
        if not isinstance(content, str):
            raise ModelClientError(
                f"Unexpected native completion response structure: {response!r}"
            )

        total_prompt_tokens = _int_or_none(response.get("tokens_evaluated"))
        completion_tokens = _int_or_none(response.get("tokens_predicted"))

        cached_tokens: int | None = None
        prompt_evaluated_tokens: int | None = None
        prompt_seconds: float | None = None
        generation_seconds: float | None = None
        timings = response.get("timings")
        if isinstance(timings, dict):
            prompt_n = timings.get("prompt_n")
            prompt_ms = timings.get("prompt_ms")
            predicted_n = timings.get("predicted_n")
            predicted_ms = timings.get("predicted_ms")
            if isinstance(prompt_n, int):
                prompt_evaluated_tokens = prompt_n
            if completion_tokens is None and isinstance(predicted_n, int):
                completion_tokens = predicted_n
            if isinstance(prompt_ms, (int, float)):
                prompt_seconds = float(prompt_ms) / 1000.0
            if isinstance(predicted_ms, (int, float)):
                generation_seconds = float(predicted_ms) / 1000.0

        # Native /completion's tokens_cached is not the same metric as the
        # OpenAI-compatible timings.cache_n that we used previously. For a
        # directly comparable "reused prompt prefix" metric, derive it from
        # total prompt tokens minus the suffix actually evaluated this request.
        if total_prompt_tokens is not None and prompt_evaluated_tokens is not None:
            cached_tokens = max(total_prompt_tokens - prompt_evaluated_tokens, 0)

        # If tokens_evaluated is unavailable, timings.prompt_n still gives the
        # amount of prompt work done, but the total prompt length is unknown.
        if total_prompt_tokens is None:
            total_prompt_tokens = prompt_evaluated_tokens

        if self.trace_prompt_reuse and prompt_token_ids is not None:
            generated_token_ids = _token_id_list(response.get("tokens"))
            if expected_common is None:
                LOGGER.info(
                    "prompt-reuse slot=%s first-observed prompt=%d generated=%d",
                    self.id_slot if self.id_slot is not None else "auto",
                    len(prompt_token_ids),
                    len(generated_token_ids),
                )
            else:
                expected_new = len(prompt_token_ids) - expected_common
                LOGGER.info(
                    "prompt-reuse slot=%s resident=%d prompt=%d common=%d expected-new=%d actual-new=%s",
                    self.id_slot if self.id_slot is not None else "auto",
                    resident_before,
                    len(prompt_token_ids),
                    expected_common,
                    expected_new,
                    prompt_evaluated_tokens
                    if prompt_evaluated_tokens is not None
                    else "?",
                )
            # The next prompt can reuse both the evaluated prompt and tokens
            # generated by this completion, because both now live in the slot.
            self._slot_tokens = prompt_token_ids + generated_token_ids

        return ChatResponse(
            content=content,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_seconds=elapsed,
            cached_tokens=cached_tokens,
            prompt_evaluated_tokens=prompt_evaluated_tokens,
            prompt_seconds=prompt_seconds,
            generation_seconds=generation_seconds,
        )

    def _apply_template(self, messages: list[dict[str, str]]) -> str:
        response = self._post_json(
            self.apply_template_url,
            {"messages": messages},
        )
        prompt = response.get("prompt")
        if not isinstance(prompt, str):
            raise ModelClientError(
                f"Unexpected /apply-template response structure: {response!r}"
            )
        return prompt

    def _tokenize(self, prompt: str) -> list[int]:
        response = self._post_json(
            self.tokenize_url,
            {
                "content": prompt,
                "add_special": True,
                "parse_special": True,
                "with_pieces": False,
            },
        )
        tokens = _token_id_list(response.get("tokens"))
        if not tokens:
            raise ModelClientError(
                f"Unexpected /tokenize response structure: {response!r}"
            )
        return tokens

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


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _token_id_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [token for token in value if isinstance(token, int)]


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    common = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        common += 1
    return common
