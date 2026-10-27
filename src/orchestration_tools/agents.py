"""Independent native Conversation agents sharing one Engine with the manager.

All run() calls must be serialized by the CORE's single model executor.
This module deliberately does not import the old orchestration package.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

LOGGER = logging.getLogger(__name__)


class AgentState(str, Enum):
    FREE = "FREE"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AgentOutcome:
    agent_id: str
    state: AgentState
    text: str
    kv_before_reset: int | None


class NativeAgent:
    """One immutable prewarmed BASE and disposable working dialogue."""

    def __init__(self, agent_id: str, chat):
        if not agent_id.strip():
            raise ValueError("agent_id must not be empty")
        self.agent_id = agent_id
        self.chat = chat
        self.state = AgentState.FREE
        self._closed = False

    @property
    def base_kv_tokens(self) -> int | None:
        return self.chat.base_kv_tokens

    def run(self, task: str) -> AgentOutcome:
        if self._closed or self.state is not AgentState.FREE:
            raise RuntimeError(f"{self.agent_id} is not available: {self.state.value}")
        if not task or not task.strip():
            raise ValueError("Agent task must not be empty")
        # A completed native send_message can include multiple tool calls.
        self.state = AgentState.RUNNING
        content = ""
        failure: BaseException | None = None
        kv = None
        try:
            content = self.chat.ask(task)
            kv = self.chat.working_kv_tokens
            if not content.strip():
                failure = RuntimeError("Agent returned an empty response")
        except Exception as exc:
            failure = exc
            kv = self.chat.working_kv_tokens
            LOGGER.exception("%s task failed", self.agent_id)
        finally:
            # Never reuse task history. A new clone is made from this AGENT's BASE.
            try:
                self.chat.new_dialogue()
            except Exception as exc:
                self.state = AgentState.FAILED
                LOGGER.exception("%s failed to reset its working context", self.agent_id)
                raise RuntimeError(f"{self.agent_id} reset failed: {exc}") from exc
            if not self._closed:
                self.state = AgentState.FREE
        if failure is not None:
            return AgentOutcome(self.agent_id, AgentState.FAILED, str(failure), kv)
        return AgentOutcome(self.agent_id, AgentState.FREE, content, kv)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.chat.close()


class AgentPool:
    """Fixed per-agent BASEs; explicit worker selection; no implicit engine parallelism."""

    def __init__(self, agents: list[NativeAgent]):
        self._agents: dict[str, NativeAgent] = {}
        for agent in agents:
            if agent.agent_id in self._agents:
                raise ValueError(f"Duplicate agent id: {agent.agent_id}")
            self._agents[agent.agent_id] = agent
        self._closed = False

    def status(self) -> list[dict[str, object]]:
        return [
            {"agent_id": a.agent_id, "state": a.state.value,
             "base_kv_tokens": a.base_kv_tokens}
            for a in self._agents.values()
        ]

    def run(self, task: str, agent_id: str) -> AgentOutcome:
        if self._closed:
            raise RuntimeError("Agent pool is closed")
        try:
            agent = self._agents[agent_id]
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {agent_id}") from exc
        return agent.run(task)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for agent in self._agents.values():
            agent.close()
