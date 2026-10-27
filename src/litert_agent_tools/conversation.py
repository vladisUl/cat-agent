"""Independent Conversation lifecycle, separate from src/litert_agent's Session code."""

from __future__ import annotations

from collections.abc import Mapping
import logging
import time
from typing import Any

from .clone import clone_conversation
from .native_json_trace import attach_native_json_trace


LOGGER = logging.getLogger(__name__)


class InvalidUserTextError(ValueError):
    """Reject terminal-corrupted Unicode before entering the native runtime."""


def _validate_user_text(text: str) -> None:
    """Prevent lone UTF-16 surrogate escapes from reaching LiteRT's JSON parser."""
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        codepoint = ord(text[exc.start])
        LOGGER.warning(
            "native-conversation REJECT-INPUT invalid_unicode=U+%04X offset=%d; "
            "resident KV was not touched",
            codepoint, exc.start,
        )
        raise InvalidUserTextError(
            f"Invalid Unicode U+{codepoint:04X} at character {exc.start}; "
            "re-enter the request (terminal UTF-8 input was corrupted)."
        ) from exc


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
    try:
        info = reader()
    except Exception:
        LOGGER.debug("Native benchmark info is unavailable", exc_info=True)
        return None
    value = getattr(info, "last_prefill_token_count", None)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def response_text(response: Any) -> str:
    """Extract the final assistant text from LiteRT's Message object."""
    content = response.get("content") if isinstance(response, Mapping) else getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item if isinstance(item, str)
            else item["text"] if isinstance(item, Mapping)
                 and item.get("type") == "text"
                 and isinstance(item.get("text"), str)
            else ""
            for item in content
        )
    if content is None:
        return ""
    raise TypeError(f"Unsupported LiteRT response content: {type(content).__name__}")


class NativeConversationChat:
    """One template/base Conversation and one disposable working clone.

    One-time warm turn fills native KV using the stock Conversation API.
    Then the immutable BASE is cloned for real dialogues and /new resets.
    The BASE contains the single service exchange warm -> model reply.
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
        warm_message: str = "warm",
        trace_native_json: bool = False,
    ):
        if not system_prompt.strip():
            raise ValueError("The temporary system prompt must not be empty")
        if not warm_message.strip():
            raise ValueError("The warm message must not be empty")
        self._engine = engine
        self._base = None
        self._working = None
        self._closed = False
        self._base_kv_tokens: int | None = None
        self._base_prefilled = False
        self._trace_native_json = trace_native_json
        self._base_warm_reply: str | None = None
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

        try:
            started = time.monotonic()
            self._base = engine.create_conversation(**kwargs)
            initial_kv = _kv_tokens(self._base)

            # First real inference populates the system + tools preface KV.
            # Unlike a manually rendered prefill, the stock 0.18.x API
            # necessarily retains this user/assistant exchange in BASE.
            response = self._base.send_message(warm_message)
            self._base_warm_reply = response_text(response)
            if not self._base_warm_reply.strip():
                raise RuntimeError("BASE warm message returned an empty assistant reply")

            self._base_kv_tokens = _kv_tokens(self._base)
            if self._base_kv_tokens is None or self._base_kv_tokens <= 0:
                raise RuntimeError(
                    "BASE warm response completed but native KV token_count "
                    "is missing or empty; refusing to clone a cold BASE"
                )
            if initial_kv is not None and self._base_kv_tokens <= initial_kv:
                raise RuntimeError(
                    "BASE warm response did not increase native KV token_count"
                )
            self._base_prefilled = True
            LOGGER.info(
                "native-conversation BASE-WARM message=%r response=%r "
                "kv_before=%s kv_after=%d elapsed=%.3fs",
                warm_message, self._base_warm_reply, initial_kv,
                self._base_kv_tokens, time.monotonic() - started,
            )

            # The BASE is immutable from this point forward.
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

    @property
    def base_warm_reply(self) -> str | None:
        return self._base_warm_reply

    def _clone_base(self):
        assert self._base is not None
        fresh = clone_conversation(self._base)
        if self._trace_native_json:
            fresh = attach_native_json_trace(fresh)
        kv = _kv_tokens(fresh)
        LOGGER.info("native-conversation BASE-CLONE kv_tokens=%s expected=%s", kv, self._base_kv_tokens)
        if kv != self._base_kv_tokens:
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
        _validate_user_text(question)
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
        text = response_text(result)
        if not text.strip():
            LOGGER.warning(
                "native-conversation EMPTY-REPLY response_type=%s response=%r",
                type(result).__name__, repr(result)[:2048],
            )
        return text

    def new_dialogue(self) -> None:
        """Drop user dialogue; preserve the already-prefilled warm exchange."""
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
