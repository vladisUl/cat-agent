import logging

from .core_scheduler import CoreScheduler, MANAGER_PRIORITY

LOGGER = logging.getLogger(__name__)

VOICE_PRIORITY = MANAGER_PRIORITY - 10
VOICE_REQUEST_LABEL = "voice"


class VoiceCoreScheduler(CoreScheduler):
    def _prepare_contexts(self):
        super()._prepare_contexts()
        fork = getattr(self.bundle.runtime, "fork_context", None)
        if "voice" not in self._contexts and callable(fork):
            context = fork("voice")
            self._contexts["voice"] = context
            context.client.set_event_handler(self._model_event)
            LOGGER.info("CORE voice session base ready")

    def submit_voice(self, text, *, session_id="", request_id=None):
        return self._submit_input(text, VOICE_REQUEST_LABEL, VOICE_PRIORITY, session_id, request_id)
