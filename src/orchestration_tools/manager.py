"""Native Conversation manager: one active dialogue, no legacy protocol.

CORE guarantees serialized calls. This object owns only the model resources and
returns structured turns; transport concerns stay in agent_core_tools.
"""
from __future__ import annotations

import logging

from .contracts import ManagerTurn
from .agents import AgentOutcome, AgentPool

LOGGER = logging.getLogger(__name__)


class NativeManager:
    def __init__(self, chat, engine, *, model_name: str, agents: AgentPool | None = None):
        self.chat = chat
        self.engine = engine
        self.model_name = model_name
        self._closed = False
        self.agents = agents if agents is not None else AgentPool([])

    @property
    def base_kv_tokens(self):
        return self.chat.base_kv_tokens

    def respond(self, session_id: str, text: str) -> ManagerTurn:
        if self._closed:
            raise RuntimeError("Manager is closed")
        answer = self.chat.ask(text)
        return ManagerTurn("reply", answer) if answer.strip() else ManagerTurn("silent")

    def agent_status(self) -> list[dict[str, object]]:
        return self.agents.status()

    def run_agent(self, agent_id: str, task: str) -> AgentOutcome:
        if self._closed:
            raise RuntimeError("Manager is closed")
        return self.agents.run(task, agent_id)

    def release_session(self, session_id: str) -> None:
        if not self._closed:
            # Do not reuse the last owner's dialogue in a new session.
            self.chat.new_dialogue()
            LOGGER.info("Native manager session released; BASE clone restored")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.agents.close()
        finally:
            try:
                self.chat.close()
            finally:
                self.engine.close()
