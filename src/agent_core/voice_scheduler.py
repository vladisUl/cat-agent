from .core_scheduler import CoreScheduler, MANAGER_PRIORITY

VOICE_PRIORITY = MANAGER_PRIORITY - 10
VOICE_REQUEST_LABEL = "voice"


class VoiceCoreScheduler(CoreScheduler):
    def submit_voice(self, text, *, session_id="", request_id=None):
        return self._submit_input(text, VOICE_REQUEST_LABEL, VOICE_PRIORITY, session_id, request_id)
