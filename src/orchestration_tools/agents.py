"""Independent native Conversation agents sharing one Engine with the manager.

All run() calls must be serialized by the CORE's single model executor.
This module deliberately does not import the old orchestration package.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging

LOGGER = logging.getLogger(__name__)

# A no-output, no-tool turn has no observable task result. One corrective
# prompt in the same working dialogue is safe: no native tool side effect
# has occurred, so this cannot replay a previously completed action.
NO_ACTION_CORRECTION = (
    "Предыдущий ответ был пустым, и ни один инструмент не был вызван. "
    "Задание ещё не выполнено. Выполни исходное задание сейчас: "
    "вызови необходимые доступные инструменты. "
    "Если пользователь просил сообщить результат, верни фактические данные текстом. "
    "Если задача не может быть выполнена, объясни причину вместо пустого ответа."
)


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
    status_code: int = 0


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
            # A successful silent turn requires an actual native tool call.
            # A model that produces neither a result nor a tool invocation
            # has not attempted the task. Correct it once, without replaying
            # any side effects: this branch is reachable only with 0 tools.
            calls = getattr(self.chat, "execution_tool_calls", None)
            if not content.strip() and calls == 0:
                LOGGER.warning("%s no-action turn (0 tools, empty text); "
                               "requesting one correction", self.agent_id)
                content = self.chat.ask(NO_ACTION_CORRECTION)
                kv = self.chat.working_kv_tokens
                calls = getattr(self.chat, "execution_tool_calls", None)
            if getattr(self.chat, "execution_status_code", 0):
                failure = RuntimeError("One or more native tools failed")
            elif not content.strip() and calls == 0:
                failure = RuntimeError("Agent did not execute tools or return a result")
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
            return AgentOutcome(self.agent_id, AgentState.FAILED, str(failure), kv, 1)
        return AgentOutcome(self.agent_id, AgentState.FREE, content, kv, 0)

    def run_slice(self, instruction: str, *, final: bool) -> AgentOutcome:
        """One non-preemptible native model call for a cooperative background job.

        WAITING keeps the same working Conversation resident between slices.
        The caller must execute this only on the CORE model worker.
        """
        if self._closed or self.state not in (AgentState.FREE, AgentState.WAITING):
            raise RuntimeError(f"{self.agent_id} is not available: {self.state.value}")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Background slice must not be empty")
        self.state = AgentState.RUNNING
        failure = None
        text = ""
        kv = None
        try:
            text = self.chat.ask(instruction)
            kv = self.chat.working_kv_tokens
            if getattr(self.chat, "execution_status_code", 0):
                failure = RuntimeError("One or more native tools failed")
        except Exception as exc:
            failure = exc
            kv = self.chat.working_kv_tokens
            LOGGER.exception("%s background slice failed", self.agent_id)
        if failure is not None or final:
            try:
                self.chat.new_dialogue()
            except Exception as exc:
                self.state = AgentState.FAILED
                LOGGER.exception("%s background context reset failed", self.agent_id)
                raise RuntimeError(f"{self.agent_id} reset failed: {exc}") from exc
            self.state = AgentState.FREE
        else:
            self.state = AgentState.WAITING
        if failure is not None:
            return AgentOutcome(self.agent_id, AgentState.FAILED, str(failure), kv, 1)
        return AgentOutcome(self.agent_id, AgentState.FREE if final else AgentState.WAITING, text, kv, 0)

    def abort_slices(self) -> None:
        """Discard a suspended task without replaying the previous native call."""
        if self._closed:
            return
        if self.state is AgentState.RUNNING:
            raise RuntimeError(f"{self.agent_id} is still running")
        if self.state is AgentState.WAITING:
            try:
                self.chat.new_dialogue()
            except Exception as exc:
                self.state = AgentState.FAILED
                raise RuntimeError(f"{self.agent_id} reset failed: {exc}") from exc
            self.state = AgentState.FREE

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

    def run_slice(self, agent_id: str, text: str, *, final: bool) -> AgentOutcome:
        if self._closed:
            raise RuntimeError("Agent pool is closed")
        try:
            return self._agents[agent_id].run_slice(text, final=final)
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {agent_id}") from exc

    def abort_slices(self, agent_id: str) -> None:
        if self._closed:
            return
        try:
            self._agents[agent_id].abort_slices()
        except KeyError as exc:
            raise ValueError(f"Unknown agent: {agent_id}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for agent in self._agents.values():
            agent.close()
