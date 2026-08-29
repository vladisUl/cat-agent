from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import time
from typing import Any, Callable
from urllib import error, request

from orchestration.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WarmResult:
    strategy: str
    elapsed_seconds: float
    token_count: int


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    phase: str
    phase_started: float | None
    prefill_seconds: float | None
    generation_seconds: float | None
    total_seconds: float | None
    finished_at: float | None


ModelEventHandler = Callable[[str, str, str], None]


class LlamaChatClient:
    """llama-server native API adapter for the current cat-agent CORE."""

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
        id_slot: int,
    ) -> None:
        base = api_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self.api_base_url = base.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.label = label
        self.id_slot = id_slot

        self._base_messages: list[dict[str, str]] = []
        self._base_resident_tokens = 0
        self._resident_tokens = 0
        self._last_response: ChatResponse | None = None
        self._event_handler: ModelEventHandler | None = None
        self._inference_timing = InferenceTiming("idle", None, None, None, None, None)

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
        while not stop_requested():
            try:
                response = self._send_json(request.Request(self.health_url, method="GET"))
                if response.get("status") != "ok":
                    raise ModelClientError(
                        f"Unexpected health response from llama-server: {response!r}"
                    )
                LOGGER.info("llama-server is ready")
                return True
            except ModelClientError as exc:
                LOGGER.info("Waiting for llama-server: %s", exc)
                time.sleep(interval)
        return False

    def close(self) -> None:
        self._base_messages = []
        self._base_resident_tokens = 0
        self._resident_tokens = 0
        self._last_response = None
        self._event_handler = None
        self._inference_timing = InferenceTiming("idle", None, None, None, None, None)

    def prepare_prefix(self, messages: list[dict[str, str]]) -> WarmResult:
        if (
            len(messages) != 1
            or messages[0].get("role") != "system"
            or not messages[0].get("content", "").strip()
        ):
            raise ModelClientError(
                "llama.cpp base must be exactly one non-empty system message"
            )

        prompt = self._apply_template(messages)
        token_ids = self._tokenize(prompt)
        started = time.monotonic()
        response = self._completion_request(prompt, n_predict=0, stream=False)
        elapsed = time.monotonic() - started

        token_count = _int_or_none(response.get("tokens_evaluated")) or len(token_ids)
        if token_count <= 0:
            raise ModelClientError("llama.cpp prefix warmup evaluated no tokens")

        self._base_messages = deepcopy(messages)
        self._base_resident_tokens = token_count
        self._resident_tokens = token_count
        self._last_response = None
        self._inference_timing = InferenceTiming(
            "idle",
            None,
            _timing_seconds(response, "prompt_ms"),
            0.0,
            elapsed,
            time.monotonic(),
        )
        LOGGER.info(
            "llama-slot-warm %s slot=%d resident=%d elapsed=%.3fs",
            self.label,
            self.id_slot,
            token_count,
            elapsed,
        )
        return WarmResult("cache-prompt", elapsed, token_count)

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        if (
            len(messages) != 1
            or messages[0].get("role") != "system"
            or not messages[0].get("content", "").strip()
        ):
            raise ModelClientError(
                "llama.cpp reset base must be exactly one non-empty system message"
            )

        # llama-server cache_prompt compares the complete prompt on the next
        # request and discards a divergent tail itself. No extra model request
        # is needed here. Keep logical telemetry at BASE immediately.
        if messages == self._base_messages:
            base_tokens = self._base_resident_tokens
        else:
            self._base_messages = deepcopy(messages)
            base_tokens = 0
            self._base_resident_tokens = 0
        self._resident_tokens = base_tokens
        self._last_response = None
        LOGGER.info(
            "llama-slot-reset %s slot=%d resident=%d",
            self.label,
            self.id_slot,
            self._resident_tokens,
        )

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        if not messages or messages[-1].get("role") != "user":
            raise ModelClientError("llama.cpp turn must end in a user message")
        return self._complete(messages, n_predict=self.max_output_tokens)

    def prefill(self, messages: list[dict[str, str]]) -> ChatResponse:
        return self._complete(messages, n_predict=0)

    def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        n_predict: int,
    ) -> ChatResponse:
        total_started = time.monotonic()
        prompt = self._apply_template(messages)
        stream = self._event_handler is not None and n_predict > 0

        prompt_started = time.monotonic()
        prompt_seconds: float | None = None
        generation_seconds: float | None = None
        self._inference_timing = InferenceTiming(
            "prefill", prompt_started, None, None, None, None
        )
        self._emit_event("prefill_start")

        try:
            if stream:
                response, content, first_chunk_at = self._completion_stream(
                    prompt,
                    n_predict=n_predict,
                )
                if first_chunk_at is not None:
                    prompt_seconds = first_chunk_at - prompt_started
            else:
                response = self._completion_request(
                    prompt,
                    n_predict=n_predict,
                    stream=False,
                )
                content = response.get("content")
                if not isinstance(content, str):
                    raise ModelClientError(
                        f"Unexpected llama-server completion response: {response!r}"
                    )
                self._emit_event("chunk", content)

            elapsed = time.monotonic() - total_started
            exact_prompt = _timing_seconds(response, "prompt_ms")
            exact_generation = _timing_seconds(response, "predicted_ms")
            if exact_prompt is not None:
                prompt_seconds = exact_prompt
            if exact_generation is not None:
                generation_seconds = exact_generation
            elif prompt_seconds is not None:
                generation_seconds = max(elapsed - prompt_seconds, 0.0)

            total_prompt_tokens = _int_or_none(response.get("tokens_evaluated"))
            completion_tokens = _int_or_none(response.get("tokens_predicted"))
            timings = response.get("timings")
            if isinstance(timings, dict):
                if total_prompt_tokens is None:
                    total_prompt_tokens = _int_or_none(timings.get("prompt_n"))
                if completion_tokens is None:
                    completion_tokens = _int_or_none(timings.get("predicted_n"))

            cached_tokens = _int_or_none(response.get("tokens_cached"))
            prompt_evaluated_tokens: int | None = None
            if total_prompt_tokens is not None and cached_tokens is not None:
                prompt_evaluated_tokens = max(total_prompt_tokens - cached_tokens, 0)
            elif isinstance(timings, dict):
                prompt_evaluated_tokens = _int_or_none(timings.get("prompt_n"))
                if total_prompt_tokens is not None and prompt_evaluated_tokens is not None:
                    cached_tokens = max(total_prompt_tokens - prompt_evaluated_tokens, 0)

            if total_prompt_tokens is not None:
                self._resident_tokens = total_prompt_tokens + (completion_tokens or 0)

            finished_at = time.monotonic()
            self._inference_timing = InferenceTiming(
                "idle",
                None,
                prompt_seconds,
                generation_seconds,
                elapsed,
                finished_at,
            )
            self._emit_event("decode_done")

            result = ChatResponse(
                content=content,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=completion_tokens,
                elapsed_seconds=elapsed,
                cached_tokens=cached_tokens,
                prompt_evaluated_tokens=prompt_evaluated_tokens,
                prompt_seconds=prompt_seconds,
                generation_seconds=generation_seconds,
            )
            self._last_response = result
            return result
        except ModelClientError:
            self._finish_failed_timing(total_started, prompt_seconds, generation_seconds)
            raise
        except Exception as exc:
            self._finish_failed_timing(total_started, prompt_seconds, generation_seconds)
            raise ModelClientError(f"llama-server request failed: {exc}") from exc

    def _completion_stream(
        self,
        prompt: str,
        *,
        n_predict: int,
    ) -> tuple[dict[str, Any], str, float | None]:
        payload = self._completion_payload(prompt, n_predict=n_predict, stream=True)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            self.completion_url,
            data=encoded,
            method="POST",
            headers={"Content-Type": "application/json"},
        )

        chunks: list[str] = []
        final: dict[str, Any] | None = None
        first_chunk_at: float | None = None
        decode_started = False

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                for raw in response:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    item = json.loads(data)
                    if not isinstance(item, dict):
                        continue

                    chunk = item.get("content")
                    if isinstance(chunk, str) and chunk:
                        if not decode_started:
                            first_chunk_at = time.monotonic()
                            self._inference_timing = InferenceTiming(
                                "generate",
                                first_chunk_at,
                                None,
                                None,
                                None,
                                None,
                            )
                            self._emit_event("decode_start")
                            decode_started = True
                        chunks.append(chunk)
                        self._emit_event("chunk", chunk)

                    if item.get("stop") is True:
                        final = item
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelClientError(
                f"HTTP {exc.code} from llama-server: {body}"
            ) from exc
        except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"llama-server stream failed: {exc}") from exc

        if final is None:
            raise ModelClientError("llama-server stream ended without final response")
        return final, "".join(chunks), first_chunk_at

    def _completion_request(
        self,
        prompt: str,
        *,
        n_predict: int,
        stream: bool,
    ) -> dict[str, Any]:
        return self._post_json(
            self.completion_url,
            self._completion_payload(prompt, n_predict=n_predict, stream=stream),
        )

    def _completion_payload(
        self,
        prompt: str,
        *,
        n_predict: int,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "stream": stream,
            "n_predict": n_predict,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cache_prompt": True,
            "id_slot": self.id_slot,
        }

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
        tokens = response.get("tokens")
        if not isinstance(tokens, list):
            raise ModelClientError(
                f"Unexpected /tokenize response structure: {response!r}"
            )
        token_ids = [token for token in tokens if isinstance(token, int)]
        if not token_ids:
            raise ModelClientError("llama-server tokenized prefix is empty")
        return token_ids

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        http_request = request.Request(
            url,
            data=encoded,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._send_json(http_request)

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
                message = f"HTTP {exc.code} from llama-server: {body}"
                if 400 <= exc.code < 500:
                    raise ModelClientError(message) from exc
                last_error = ModelClientError(message)
            except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc

            if attempt < attempts:
                delay = self.retry_delay_seconds * attempt
                LOGGER.warning(
                    "llama-server request failed (%s/%s): %s; retry in %.1f s",
                    attempt,
                    attempts,
                    last_error,
                    delay,
                )
                time.sleep(delay)

        raise ModelClientError(f"llama-server request failed: {last_error}")

    def _finish_failed_timing(
        self,
        total_started: float,
        prompt_seconds: float | None,
        generation_seconds: float | None,
    ) -> None:
        now = time.monotonic()
        current = self._inference_timing
        if prompt_seconds is None and current.phase == "prefill" and current.phase_started is not None:
            prompt_seconds = now - current.phase_started
        if generation_seconds is None and current.phase == "generate" and current.phase_started is not None:
            generation_seconds = now - current.phase_started
        self._inference_timing = InferenceTiming(
            "idle",
            None,
            prompt_seconds,
            generation_seconds,
            now - total_started,
            now,
        )
        self._emit_event("decode_error")

    def _emit_event(self, event: str, payload: str = "") -> None:
        handler = self._event_handler
        if handler is None:
            return
        try:
            handler(self.label, event, payload)
        except Exception:
            LOGGER.debug(
                "model event handler failed label=%s event=%s",
                self.label,
                event,
                exc_info=True,
            )


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _timing_seconds(response: dict[str, Any], field: str) -> float | None:
    timings = response.get("timings")
    if not isinstance(timings, dict):
        return None
    value = timings.get(field)
    if not isinstance(value, (int, float)):
        return None
    return float(value) / 1000.0
