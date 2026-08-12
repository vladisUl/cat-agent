from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import inspect
import logging
import time
from typing import Any, Callable

import litert_lm

from cat_agent.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WarmResult:
    strategy: str
    elapsed_seconds: float
    token_count: int


class LiteRT015ChatClient:
    """Stateful LiteRT-LM 0.15 adapter used only for the comparison benchmark.

    The goal is to keep the production 0.14 path untouched while extracting a
    fair four-pass timing profile from the already installed 0.15 runtime.
    Prefix preparation tries, in order:

    1. native prefill_preface_on_init if the installed Python API exposes it;
    2. normal generation of the canonical bootstrap ACK (exact history);
    3. the previously tested max_output_tokens=0 assistant-ACK warmup;
    4. a tiny hidden warmup turn as a last-resort timing approximation.

    Strategies 1-3 preserve the canonical logical history exactly. Strategy 4
    is deliberately labelled "shim" in the output so it cannot be mistaken for
    an exact semantic prefill.
    """

    def __init__(
        self,
        engine: litert_lm.Engine,
        *,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
        label: str,
    ) -> None:
        self.engine = engine
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.label = label
        self.sampler_config = litert_lm.SamplerConfig(
            top_p=top_p,
            temperature=temperature,
            seed=0,
        )
        self.thinking_config = litert_lm.ThinkingConfig(
            enable_thinking=False,
            thinking_token_budget=0,
        )
        self._conversation: litert_lm.Conversation | None = None
        self._synced_messages: list[dict[str, str]] = []

    def wait_until_ready(
        self, stop_requested: Callable[[], bool], interval: float = 2.0
    ) -> bool:
        del interval
        return not stop_requested()

    def close(self) -> None:
        if self._conversation is not None:
            self._conversation.close()
            self._conversation = None
        self._synced_messages = []

    def prepare_prefix(self, messages: list[dict[str, str]]) -> WarmResult:
        if len(messages) < 3 or messages[-1].get("role") != "assistant":
            raise ModelClientError(
                "LiteRT 0.15 prefix preparation expects canonical history ending in assistant ACK"
            )
        expected = messages[-1].get("content")
        if not isinstance(expected, str) or not expected:
            raise ModelClientError("Canonical assistant ACK must be non-empty")

        native = self._try_native_prefill(messages)
        if native is not None:
            return native

        generated = self._try_generated_ack(messages, expected)
        if generated is not None:
            return generated

        forced = self._try_forced_ack(messages, expected)
        if forced is not None:
            return forced

        return self._shim_warm(messages)

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        if not messages or messages[-1].get("role") != "user":
            raise ModelClientError(
                "LiteRT 0.15 adapter expects each model turn to end in a user message"
            )

        try:
            if not self._history_extends_resident(messages):
                self._reset_conversation(messages)

            suffix = messages[len(self._synced_messages) :]
            if len(suffix) != 1 or suffix[0].get("role") != "user":
                self._reset_conversation(messages)
                suffix = messages[len(self._synced_messages) :]

            assert self._conversation is not None
            resident_before = self._conversation.token_count
            started = time.monotonic()
            response = self._conversation.send_message(
                suffix[0],
                max_output_tokens=self.max_output_tokens,
                thinking_config=self.thinking_config,
            )
            elapsed = time.monotonic() - started

            content = _response_text(response)
            benchmark = self._conversation.get_benchmark_info()
            resident_after = self._conversation.token_count

            prefill_n = benchmark.last_prefill_token_count
            decode_n = benchmark.last_decode_token_count
            prompt_seconds = _seconds_from_rate(
                prefill_n, benchmark.last_prefill_tokens_per_second
            )
            generation_seconds = _seconds_from_rate(
                decode_n, benchmark.last_decode_tokens_per_second
            )

            self._synced_messages = deepcopy(messages)
            self._synced_messages.append(
                {"role": "assistant", "content": content}
            )

            LOGGER.info(
                "litert015-kv %s resident=%d new=%d decode=%d after=%d wall=%.3fs "
                "prefill=%s generate=%s",
                self.label,
                resident_before,
                prefill_n,
                decode_n,
                resident_after,
                elapsed,
                _fmt_seconds(prompt_seconds),
                _fmt_seconds(generation_seconds),
            )

            return ChatResponse(
                content=content,
                prompt_tokens=resident_before + prefill_n,
                completion_tokens=decode_n,
                elapsed_seconds=elapsed,
                cached_tokens=resident_before,
                prompt_evaluated_tokens=prefill_n,
                prompt_seconds=prompt_seconds,
                generation_seconds=generation_seconds,
            )
        except ModelClientError:
            raise
        except Exception as exc:
            raise ModelClientError(f"LiteRT-LM 0.15 request failed: {exc}") from exc

    def _try_native_prefill(
        self, messages: list[dict[str, str]]
    ) -> WarmResult | None:
        try:
            params = inspect.signature(self.engine.create_conversation).parameters
        except (TypeError, ValueError):
            return None
        if "prefill_preface_on_init" not in params:
            LOGGER.info("litert015-warm %s native-prefill unavailable", self.label)
            return None

        try:
            self.close()
            started = time.monotonic()
            self._conversation = self.engine.create_conversation(
                messages=deepcopy(messages),
                automatic_tool_calling=False,
                sampler_config=self.sampler_config,
                thinking_config=self.thinking_config,
                prefill_preface_on_init=True,
                max_output_tokens=self.max_output_tokens,
            )
            elapsed = time.monotonic() - started
            resident = self._conversation.token_count
            if resident <= 0:
                self.close()
                return None
            self._synced_messages = deepcopy(messages)
            LOGGER.info(
                "litert015-warm %s strategy=native resident=%d elapsed=%.3fs",
                self.label,
                resident,
                elapsed,
            )
            return WarmResult("native", elapsed, resident)
        except Exception as exc:
            LOGGER.warning(
                "litert015-warm %s native-prefill rejected: %s", self.label, exc
            )
            self.close()
            return None

    def _try_generated_ack(
        self, messages: list[dict[str, str]], expected: str
    ) -> WarmResult | None:
        # For canonical [system, user bootstrap, assistant READY], create the
        # Conversation with the stable history before the bootstrap user turn,
        # then let the model generate the exact canonical ACK normally.
        base = deepcopy(messages[:-2])
        bootstrap_user = deepcopy(messages[-2])
        if bootstrap_user.get("role") != "user":
            return None

        try:
            self.close()
            self._conversation = self._create_conversation(base)
            started = time.monotonic()
            response = self._conversation.send_message(
                bootstrap_user,
                max_output_tokens=self.max_output_tokens,
                thinking_config=self.thinking_config,
            )
            elapsed = time.monotonic() - started
            content = _response_text(response).strip()
            resident = self._conversation.token_count
            if content != expected:
                LOGGER.warning(
                    "litert015-warm %s generated ACK mismatch expected=%r got=%r",
                    self.label,
                    expected,
                    content,
                )
                self.close()
                return None

            self._synced_messages = deepcopy(messages)
            LOGGER.info(
                "litert015-warm %s strategy=generated resident=%d elapsed=%.3fs",
                self.label,
                resident,
                elapsed,
            )
            return WarmResult("generated", elapsed, resident)
        except Exception as exc:
            LOGGER.warning(
                "litert015-warm %s generated ACK failed: %s", self.label, exc
            )
            self.close()
            return None

    def _try_forced_ack(
        self, messages: list[dict[str, str]], expected: str
    ) -> WarmResult | None:
        # Historical 0.15 fallback: submit the known assistant ACK with a zero
        # output budget. Some builds preserve the ACK; others decode one token
        # such as NEED, in which case this attempt is discarded completely.
        try:
            self.close()
            self._conversation = self._create_conversation(messages[:-1])
            started = time.monotonic()
            response = self._conversation.send_message(
                deepcopy(messages[-1]),
                max_output_tokens=0,
                thinking_config=self.thinking_config,
            )
            elapsed = time.monotonic() - started
            content = _response_text(response).strip()
            resident = self._conversation.token_count
            if content != expected:
                LOGGER.warning(
                    "litert015-warm %s forced ACK mismatch expected=%r got=%r",
                    self.label,
                    expected,
                    content,
                )
                self.close()
                return None

            self._synced_messages = deepcopy(messages)
            LOGGER.info(
                "litert015-warm %s strategy=forced resident=%d elapsed=%.3fs",
                self.label,
                resident,
                elapsed,
            )
            return WarmResult("forced", elapsed, resident)
        except Exception as exc:
            LOGGER.warning(
                "litert015-warm %s forced ACK failed: %s", self.label, exc
            )
            self.close()
            return None

    def _shim_warm(self, messages: list[dict[str, str]]) -> WarmResult:
        # Last-resort benchmark-only approximation. It materializes the entire
        # canonical prefix, then adds a tiny hidden turn. The visible runtime
        # still sees the canonical history, but the underlying KV has that extra
        # warmup turn. The result is labelled "shim" so the table can flag it.
        try:
            self.close()
            self._conversation = self._create_conversation(messages)
            started = time.monotonic()
            response = self._conversation.send_message(
                {"role": "user", "content": "[warmup]"},
                max_output_tokens=0,
                thinking_config=self.thinking_config,
            )
            elapsed = time.monotonic() - started
            content = _response_text(response)
            resident = self._conversation.token_count
            self._synced_messages = deepcopy(messages)
            LOGGER.warning(
                "litert015-warm %s strategy=shim resident=%d elapsed=%.3fs hidden=%r",
                self.label,
                resident,
                elapsed,
                content,
            )
            return WarmResult("shim", elapsed, resident)
        except Exception as exc:
            self.close()
            raise ModelClientError(
                f"LiteRT-LM 0.15 could not prepare prefix by any strategy: {exc}"
            ) from exc

    def _history_extends_resident(self, messages: list[dict[str, str]]) -> bool:
        if self._conversation is None:
            return False
        if len(messages) < len(self._synced_messages):
            return False
        return messages[: len(self._synced_messages)] == self._synced_messages

    def _reset_conversation(self, messages: list[dict[str, str]]) -> None:
        if self._conversation is not None:
            LOGGER.info("litert015-kv %s reset conversation", self.label)
            self._conversation.close()
        prefix = deepcopy(messages[:-1])
        self._conversation = self._create_conversation(prefix)
        self._synced_messages = prefix
        LOGGER.info(
            "litert015-kv %s cold conversation prefix_messages=%d",
            self.label,
            len(prefix),
        )

    def _create_conversation(
        self, messages: list[dict[str, str]]
    ) -> litert_lm.Conversation:
        return self.engine.create_conversation(
            messages=deepcopy(messages),
            automatic_tool_calling=False,
            sampler_config=self.sampler_config,
            thinking_config=self.thinking_config,
            max_output_tokens=self.max_output_tokens,
        )


def _response_text(response: Any) -> str:
    if not isinstance(response, dict):
        raise ModelClientError(
            f"Unexpected LiteRT-LM 0.15 response: {response!r}"
        )

    content = response.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    raise ModelClientError(
        f"LiteRT-LM 0.15 response has no text content: {response!r}"
    )


def _seconds_from_rate(token_count: int, tokens_per_second: float) -> float | None:
    if token_count <= 0 or tokens_per_second <= 0:
        return None
    return token_count / tokens_per_second


def _fmt_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "?"
