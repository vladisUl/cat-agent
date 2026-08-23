from __future__ import annotations

import logging
import time

from .core_scheduler import CoreScheduler, MANAGER_PRIORITY, _PriorityRequest

LOGGER = logging.getLogger(__name__)

VOICE_PRIORITY = MANAGER_PRIORITY - 10
VOICE_REQUEST_LABEL = "voice"


class VoiceCoreScheduler(CoreScheduler):
    """Core scheduler extension for transient, highest-priority voice turns."""

    def submit_voice(self, text: str) -> None:
        user_text = text.strip()
        if not user_text:
            raise ValueError("voice text must be non-empty")

        with self._queue_lock:
            self._pending.append(
                _PriorityRequest(
                    kind="user",
                    label=VOICE_REQUEST_LABEL,
                    payload=user_text,
                    queued_at=time.monotonic(),
                    priority=VOICE_PRIORITY,
                )
            )

        LOGGER.info(
            "CORE queued voice request priority=%d text=%r",
            VOICE_PRIORITY,
            user_text,
        )
        self._emit_status()

    def active_request_label(self) -> str:
        request = self._active_request
        return request.label if request is not None else ""
