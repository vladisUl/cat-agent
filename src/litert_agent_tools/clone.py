"""Clone LiteRT-LM Conversations through the public Python API (0.19.0+).

The LiteRT-LM package owns its native handles and C ABI. Never call
litert_lm_conversation_clone through an application-level ctypes adapter.
"""

from __future__ import annotations

import logging


LOGGER = logging.getLogger(__name__)


class ConversationCloneError(RuntimeError):
    """The required public Conversation.clone() method is unavailable."""


def clone_conversation(source):
    """Clone the Conversation KV state, history, tools and configuration."""
    clone = getattr(source, "clone", None)
    if not callable(clone):
        raise ConversationCloneError(
            "LiteRT-LM Conversation.clone() is unavailable; install "
            "litert-lm 0.19.0+ with a matching litert-lm-api package."
        )
    result = clone()
    if result is None:
        raise ConversationCloneError("LiteRT-LM Conversation.clone() returned None")
    LOGGER.info("native-conversation CLONE mode=python success")
    return result
