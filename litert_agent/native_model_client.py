from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import logging
import time
from typing import Any, Callable

import litert_lm

from cat_agent.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NativePrefillResult:
    elapsed_seconds: float
    token_count: int


class LiteRTNativeChatClient:
    """Stateful LiteRT-LM 0.14 Conversation adapter.

    The manager and the shared neutral-agent execution path each get their own
    client instance and therefore their own persistent Conversation/KV state.
    Stable conversation prefixes are materialized directly into KV through the
    locally exposed ConversationConfig::SetPrefillPrefaceOnInit() path. No
    model-generated READY/NEED bootstrap probe is used.

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

    def prefill_prefix(
        self, messages: list[dict[str, str]]
    ) -> NativePrefillResult:
        """Materialize an exact stable conversation prefix into resident KV.

        LiteRT-LM 0.14 already has the C++
        ConversationConfig::SetPrefillPrefaceOnInit() switch. The local wheel
        exposes that switch through the C and Python bindings as
        prefill_preface_on_init=True. Conversation creation therefore performs
        prefill only: it does not ask the model to generate or validate an ACK.
        """

        if not messages:
            raise ModelClientError("LiteRT prefill_prefix requires a non-empty history")

        try:
            if self._conversation is not None:
                self._conversation.close()

            prefix = deepcopy(messages)
            started = time.monotonic()
            self._conversation = self.engine.create_conversation(
                messages=prefix,
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
                raise ModelClientError(
                    "LiteRT prefill_prefix created no resident KV tokens"
                )

            self._synced_messages = prefix
            LOGGER.info(
                "litert-kv %s prefill messages=%d resident=%d elapsed=%.3fs",
                self.label,
                len(prefix),
                resident,
                elapsed,
            )
            return NativePrefillResult(
                elapsed_seconds=elapsed,
                token_count=resident,
            )
        except ModelClientError:
            raise
        except Exception as exc:
            self.close()
            raise ModelClientError(f"LiteRT-LM native prefill failed: {exc}") from exc

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
            resident_after = self._conversation.token_count
            added = max(resident_after - resident_before, 0)

            self._synced_messages = deepcopy(messages)
            self._synced_messages.append(
                {"role": "assistant", "content": content}
            )

            # LiteRT-LM 0.14 exposes resident token_count but not the later
            # get_benchmark_info() Python API. Keep exact values exact: resident
            # KV and total tokens appended by this live turn are known; separate
            # prompt/decode counts and timings are intentionally left unknown.
            LOGGER.info(
                "litert-kv %s resident=%d added=%d after=%d wall=%.3fs",
                self.label,
                resident_before,
                added,
                resident_after,
                elapsed,
            )

            return ChatResponse(
                content=content,
                prompt_tokens=None,
                completion_tokens=None,
                elapsed_seconds=elapsed,
                cached_tokens=resident_before,
                prompt_evaluated_tokens=None,
                prompt_seconds=None,
                generation_seconds=None,
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
        started = time.monotonic()
        self._conversation = self.engine.create_conversation(
            messages=prefix,
            automatic_tool_calling=False,
            sampler_config=self.sampler_config,
            thinking_config=self.thinking_config,
            prefill_preface_on_init=bool(prefix),
            max_output_tokens=self.max_output_tokens,
        )
        elapsed = time.monotonic() - started
        resident = self._conversation.token_count
        self._synced_messages = prefix
        LOGGER.info(
            "litert-kv %s created conversation preface_messages=%d resident=%d "
            "prefill=%.3fs",
            self.label,
            len(prefix),
            resident,
            elapsed,
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
