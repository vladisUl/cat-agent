"""Independent Conversation lifecycle, separate from src/litert_agent's Session code."""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import logging
import time
from typing import Any

from .clone import clone_conversation


LOGGER = logging.getLogger(__name__)


def _kv_tokens(conversation: Any) -> int | None:
    """Native Conversation.token_count is the actual resident KV length."""
    value = getattr(conversation, "token_count", None)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _last_prefill_tokens(conversation: Any) -> int | None:
    """Last native prefill only (not the entire tool-call loop)."""
    reader = getattr(conversation, "get_benchmark_info", None)
    if not callable(reader):
        return None
    info = reader()
    value = getattr(info, "last_prefill_token_count", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def response_text(response: Any) -> str:
    """Extract the final assistant text from LiteRT's Message object."""
    content = response.get("content") if isinstance(response, Mapping) else getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item["text"] for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    if content is None:
        return ""
    raise TypeError(f"Unsupported LiteRT response content: {type(content).__name__}")


class NativeConversationChat:
    """One template/base Conversation and one disposable working clone.

    The stock LiteRT-LM 0.18.x Python/C ABI lacks the C++ prefill-preface
    switch. When a patched engine exposes the opt-in
    prefill_preface_on_init parameter, create and measure a real warmed BASE.
    Otherwise keep the existing (cold) clone behavior and report it honestly.
    """

    def __init__(
        self,
        engine: Any,
        system_prompt: str,
        tools: list,
        *,
        sampler_config: Any = None,
        max_output_tokens: int | None = None,
        thinking_config: Any = None,
    ):
        if not system_prompt.strip():
            raise ValueError("The temporary system prompt must not be empty")
        self._engine = engine
        self._base = None
        self._working = None
        self._closed = False
        self._base_kv_tokens: int | None = None
        self._base_prefilled = False
        kwargs: dict[str, Any] = {
            "messages": [{"role": "system", "content": system_prompt}],
            "tools": list(tools),
            "automatic_tool_calling": True,
        }
        if sampler_config is not None:
            kwargs["sampler_config"] = sampler_config
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        if thinking_config is not None:
            kwargs["thinking_config"] = thinking_config

        # Only the locally patched 0.18.x engine exposes this switch.
        # Do not guess a ctypes symbol that the stock C ABI does not have.
        supports_prefill = (
            "prefill_preface_on_init"
            in inspect.signature(engine.create_conversation).parameters
        )
        if supports_prefill:
            kwargs["prefill_preface_on_init"] = True

        try:
            started = time.monotonic()
            self._base = engine.create_conversation(**kwargs)
            self._base_kv_tokens = _kv_tokens(self._base)
            self._base_prefilled = bool(
                supports_prefill and self._base_kv_tokens is not None
                and self._base_kv_tokens > 0
            )
            if supports_prefill and not self._base_prefilled:
                raise RuntimeError(
                    "prefill_preface_on_init was requested, but BASE has no resident KV; "
                    "refusing to report a warmed BASE"
                )
            LOGGER.info(
                "native-conversation BASE prefill_supported=%s kv_tokens=%s elapsed=%.3fs",
                supports_prefill, self._base_kv_tokens, time.monotonic() - started,
            )
            self._working = self._clone_base()
        except BaseException:
            self.close()
            raise

    @property
    def base_prefilled(self) -> bool:
        return self._base_prefilled

    @property
    def base_kv_tokens(self) -> int | None:
        return self._base_kv_tokens

    def _clone_base(self):
        assert self._base is not None
        fresh = clone_conversation(self._base)
        kv = _kv_tokens(fresh)
        LOGGER.info("native-conversation BASE-CLONE kv_tokens=%s expected=%s", kv, self._base_kv_tokens)
        if self._base_prefilled and kv != self._base_kv_tokens:
            fresh.close()
            raise RuntimeError(
                f"Cloned KV length {kv} does not match warmed BASE {self._base_kv_tokens}"
            )
        return fresh

    def ask(self, question: str) -> str:
        if self._closed or self._working is None:
            raise RuntimeError("Conversation is closed")
        if not question.strip():
            raise ValueError("Question must not be empty")
        before = _kv_tokens(self._working)
        started = time.monotonic()
        result = self._working.send_message(question)
        after = _kv_tokens(self._working)
        last_prefill = _last_prefill_tokens(self._working)
        LOGGER.info(
            "native-conversation TURN kv_before=%s kv_after=%s "
            "last_prefill_tokens=%s elapsed=%.3fs",
            before, after, last_prefill, time.monotonic() - started,
        )
        return response_text(result)

    def new_dialogue(self) -> None:
        """Drop previous question, answers and tool results together with their KV."""
        if self._closed or self._base is None:
            raise RuntimeError("Conversation is closed")
        fresh = self._clone_base()  # Do not destroy the current session on failure.
        old, self._working = self._working, fresh
        if old is not None:
            old.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._working is not None:
                self._working.close()
        finally:
            self._working = None
            if self._base is not None:
                self._base.close()
                self._base = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
