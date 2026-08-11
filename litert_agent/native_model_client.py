from __future__ import annotations

from copy import deepcopy
import logging
import time
from typing import Any, Callable

import litert_lm

from cat_agent.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


class LiteRTNativeChatClient:
    """Stateful LiteRT-LM Conversation adapter for the existing cat-agent runtime.

    The manager and the shared neutral-agent execution path each get their own
    client instance and therefore their own persistent Conversation/KV state.
    The existing runtime still passes complete message histories; this adapter
    verifies that the new history extends the resident one and sends only the
    newly appended user turn to LiteRT-LM.
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

    def warm_prefix(self, messages: list[dict[str, str]]) -> ChatResponse:
        """Materialize a stable history ending in a known assistant message.

        LiteRT-LM 0.15 has no public Conversation prefill-only API. The tested
        equivalent is to create the Conversation with all but the final known
        assistant message, then submit that assistant message with
        max_output_tokens=0. On Gemma E4B this prefills the stable prefix and
        performs one terminal decode token while preserving correct continuation
        semantics for the next user turn.
        """

        if len(messages) < 2 or messages[-1].get("role") != "assistant":
            raise ModelClientError(
                "LiteRT warm_prefix expects a history ending in an assistant message"
            )

        expected = messages[-1].get("content", "")
        if not isinstance(expected, str) or not expected:
            raise ModelClientError("LiteRT warm_prefix requires non-empty assistant content")

        try:
            if self._conversation is not None:
                self._conversation.close()

            prefix = deepcopy(messages[:-1])
            self._conversation = self.engine.create_conversation(
                messages=prefix,
                automatic_tool_calling=False,
                sampler_config=self.sampler_config,
                thinking_config=self.thinking_config,
                max_output_tokens=self.max_output_tokens,
            )

            resident_before = self._conversation.token_count
            started = time.monotonic()
            response = self._conversation.send_message(
                deepcopy(messages[-1]),
                max_output_tokens=0,
                thinking_config=self.thinking_config,
            )
            elapsed = time.monotonic() - started

            content = _response_text(response)
            benchmark = self._conversation.get_benchmark_info()
            resident_after = self._conversation.token_count
            prefill_n = benchmark.last_prefill_token_count
            decode_n = benchmark.last_decode_token_count

            if content != expected:
                self.close()
                raise ModelClientError(
                    "LiteRT warm_prefix response mismatch: "
                    f"expected {expected!r}, got {content!r}"
                )
            if resident_after <= resident_before or prefill_n <= 0:
                self.close()
                raise ModelClientError(
                    "LiteRT warm_prefix did not materialize resident KV"
                )

            prompt_seconds = _seconds_from_rate(
                prefill_n, benchmark.last_prefill_tokens_per_second
            )
            generation_seconds = _seconds_from_rate(
                decode_n, benchmark.last_decode_tokens_per_second
            )

            # The continuation probe verified that the next real user turn can
            # continue directly from this KV state. Keep the logical history
            # aligned with cat-agent's canonical bootstrap messages.
            self._synced_messages = deepcopy(messages)

            LOGGER.info(
                "litert-kv %s warm resident=%d new=%d decode=%d after=%d "
                "prefill=%s generate=%s content=%r",
                self.label,
                resident_before,
                prefill_n,
                decode_n,
                resident_after,
                _fmt_seconds(prompt_seconds),
                _fmt_seconds(generation_seconds),
                content,
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
            self.close()
            raise ModelClientError(f"LiteRT-LM native warmup failed: {exc}") from exc

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        if not messages or messages[-1].get("role") != "user":
            raise ModelClientError(
                "LiteRT native adapter expects each model turn to end in a user message"
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

            prompt_tokens = resident_before + prefill_n
            cached_tokens = resident_before

            self._synced_messages = deepcopy(messages)
            self._synced_messages.append(
                {"role": "assistant", "content": content}
            )

            LOGGER.info(
                "litert-kv %s resident=%d new=%d prompt=%d decode=%d after=%d "
                "prefill=%s generate=%s",
                self.label,
                resident_before,
                prefill_n,
                prompt_tokens,
                decode_n,
                resident_after,
                _fmt_seconds(prompt_seconds),
                _fmt_seconds(generation_seconds),
            )

            return ChatResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=decode_n,
                elapsed_seconds=elapsed,
                cached_tokens=cached_tokens,
                prompt_evaluated_tokens=prefill_n,
                prompt_seconds=prompt_seconds,
                generation_seconds=generation_seconds,
            )
        except ModelClientError:
            raise
        except Exception as exc:
            raise ModelClientError(f"LiteRT-LM native request failed: {exc}") from exc

    def _history_extends_resident(self, messages: list[dict[str, str]]) -> bool:
        if self._conversation is None:
            return False
        if len(messages) < len(self._synced_messages):
            return False
        return messages[: len(self._synced_messages)] == self._synced_messages

    def _reset_conversation(self, messages: list[dict[str, str]]) -> None:
        if self._conversation is not None:
            LOGGER.info("litert-kv %s reset conversation", self.label)
            self._conversation.close()

        prefix = deepcopy(messages[:-1])
        self._conversation = self.engine.create_conversation(
            messages=prefix,
            automatic_tool_calling=False,
            sampler_config=self.sampler_config,
            thinking_config=self.thinking_config,
            max_output_tokens=self.max_output_tokens,
        )
        self._synced_messages = prefix
        LOGGER.info(
            "litert-kv %s created conversation with %d preface messages",
            self.label,
            len(prefix),
        )


def _response_text(response: Any) -> str:
    if not isinstance(response, dict):
        raise ModelClientError(
            f"Unexpected LiteRT-LM native response: {response!r}"
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
        f"LiteRT-LM native response has no text content: {response!r}"
    )


def _seconds_from_rate(token_count: int, tokens_per_second: float) -> float | None:
    if token_count <= 0 or tokens_per_second <= 0:
        return None
    return token_count / tokens_per_second


def _fmt_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "?"
