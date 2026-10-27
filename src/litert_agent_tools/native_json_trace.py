"""Opt-in diagnostics at LiteRT-LM 0.18 Python-to-native JSON boundary.

Does not change a native function's parameters or the model implementation.
This proxy is per working Conversation; the installed package stays intact.
"""

from __future__ import annotations

import hashlib
import json
import logging


LOGGER = logging.getLogger(__name__)


class NativeJsonTraceLib:
    """Delegate the native FFI, logging every send and details upon failure."""

    def __init__(self, lib):
        self._native = lib
        self._send_count = 0

    def __getattr__(self, name):
        if name == "litert_lm_conversation_send_message":
            return self._send_message
        return getattr(self._native, name)

    def _send_message(self, ptr, message_json, context_json, optional_args):
        self._send_count += 1
        seq = self._send_count
        try:
            wire = (
                message_json.encode("utf-8")
                if isinstance(message_json, str)
                else bytes(message_json)
            )
            parsed = json.loads(wire)
            python_valid = True
            role = (
                [item.get("role") if isinstance(item, dict) else "?"
                 for item in parsed]
                if isinstance(parsed, list)
                else parsed.get("role", "?") if isinstance(parsed, dict)
                else type(parsed).__name__
            )
            diagnostic_error = ""
        except (ValueError, TypeError, UnicodeError) as exc:
            python_valid = False
            role = "unavailable"
            diagnostic_error = str(exc)
            wire = repr(message_json).encode("utf-8", errors="replace")

        digest = hashlib.sha256(wire).hexdigest()[:16]
        LOGGER.info(
            "native-json SEND seq=%d role=%s valid_python_json=%s bytes=%d sha256=%s",
            seq, role, python_valid, len(wire), digest,
        )
        result = self._native.litert_lm_conversation_send_message(
            ptr, message_json, context_json, optional_args
        )
        if not result:
            LOGGER.error(
                "native-json FAILED seq=%d valid_python_json=%s "
                "diagnostic_error=%r raw_json=%r",
                seq, python_valid, diagnostic_error, wire[:4096],
            )
        return result


def attach_native_json_trace(conversation):
    """Trace only one cloned Conversation; other sessions are unaffected."""
    if not isinstance(conversation._lib, NativeJsonTraceLib):
        conversation._lib = NativeJsonTraceLib(conversation._lib)
    return conversation
