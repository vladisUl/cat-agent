"""Independent Conversation lifecycle, separate from src/litert_agent's Session code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .clone import clone_conversation


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

    Warning: in LiteRT-LM 0.18.x create_conversation does NOT prefill the
    system prompt. Cloning a fresh base copies the preface configuration but
    does not magically prefill its KV. Do not claim a warmed BASE until a
    supported prefill-only API has been established.
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
            self._base = engine.create_conversation(**kwargs)
            self._working = clone_conversation(self._base)
        except BaseException:
            self.close()
            raise

    @property
    def base_prefilled(self) -> bool:
        """Must stay False until a genuinely verified base-only prefill exists."""
        return False

    def ask(self, question: str) -> str:
        if self._closed or self._working is None:
            raise RuntimeError("Conversation is closed")
        if not question.strip():
            raise ValueError("Question must not be empty")
        result = self._working.send_message(question)
        return response_text(result)

    def new_dialogue(self) -> None:
        """Drop previous question, answers and tool results together with their KV."""
        if self._closed or self._base is None:
            raise RuntimeError("Conversation is closed")
        fresh = clone_conversation(self._base)  # Do not destroy the current session on failure.
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
