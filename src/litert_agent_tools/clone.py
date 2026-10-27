"""Conversation cloning for LiteRT-LM 0.18.x and newer Python bindings.

In 0.18.0/0.18.1 the C API exports
    LiteRtLmConversation* litert_lm_conversation_clone(LiteRtLmConversation*)
but Python lacks clone(). Current upstream Python provides clone() itself and
the C API uses a DIFFERENT ABI. Never apply the legacy ctypes signature to it.
"""

from __future__ import annotations

import ctypes
from importlib import metadata


class ConversationCloneError(RuntimeError):
    """Cloning is unavailable or failed."""


def clone_conversation(source):
    """Clone the native KV state, history, tool configuration and Python callbacks."""
    native = getattr(source, "clone", None)
    if callable(native):
        return native()

    try:
        version = metadata.version("litert-lm")
    except metadata.PackageNotFoundError as exc:
        raise ConversationCloneError("Cannot determine installed litert-lm version") from exc
    if version not in ("0.18.0", "0.18.1"):
        raise ConversationCloneError(
            f"Legacy Conversation.clone ctypes adapter requires LiteRT-LM 0.18.0/0.18.1, got {version}; "
            "do not guess an incompatible native ABI"
        )
    ptr = getattr(source, "_ptr", None)
    lib = getattr(source, "_lib", None)
    if not ptr or lib is None:
        raise ConversationCloneError("Source conversation has no live native handle")
    try:
        fn = lib.litert_lm_conversation_clone
    except AttributeError as exc:
        raise ConversationCloneError("Installed native library does not export Conversation.clone") from exc

    fn.argtypes = [ctypes.c_void_p]
    fn.restype = ctypes.c_void_p
    cloned_ptr = fn(ptr)
    if not cloned_ptr:
        raise ConversationCloneError("Native Conversation.clone failed (check LiteRT native logs)")

    # Import only after validating the installed ABI: tests and tooling can
    # import this module without installing LiteRT-LM.
    try:
        from litert_lm.conversation import Conversation

        return Conversation(
            lib, cloned_ptr,
            engine=source._engine,
            messages=list(source.messages),
            tools=list(source.tools),
            tools_map=dict(source._tools_map),
            tool_event_handler=source.tool_event_handler,
            automatic_tool_calling=source.automatic_tool_calling,
            extra_context=dict(source.extra_context),
            thinking_config=source.thinking_config,
            sampler_config=source.sampler_config,
            lora_config=source.lora_config,
            max_output_tokens=source.max_output_tokens,
            chat_template=source.chat_template,
            constrained_decoding_config=source.constrained_decoding_config,
        )
    except BaseException:
        lib.litert_lm_conversation_delete(cloned_ptr)
        raise
