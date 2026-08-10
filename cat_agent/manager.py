from __future__ import annotations

from dataclasses import dataclass
import logging

from .agent import AgentState
from .model_client import ModelClientError, OpenAIChatClient
from .pool import AgentPool
from .prompt_store import MANAGER_BOOTSTRAP_ACK, PromptStore
from .protocol import ManagerAction, parse_manager_output
from .skills import SkillBase, SkillBaseError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManagerTurn:
    kind: str
    text: str


class ManagerRuntime:
    def __init__(
        self,
        client: OpenAIChatClient,
        skill_base: SkillBase,
        prompt_store: PromptStore,
        pool: AgentPool,
        *,
        max_steps: int,
    ) -> None:
        self.client = client
        self.skill_base = skill_base
        self.prompt_store = prompt_store
        self.pool = pool
        self.max_steps = max_steps
        bootstrap = self._bootstrap_prompt()
        self.prompt_store.write_manager_prompt(bootstrap)
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.prompt_store.manager_system_prompt()},
            {"role": "user", "content": bootstrap},
            {"role": "assistant", "content": MANAGER_BOOTSTRAP_ACK},
        ]

    def user_message(self, text: str) -> ManagerTurn:
        user_text = text.strip()
        self.prompt_store.write_manager_prompt(f"[USER]\n{user_text}\n[/USER]")
        self._append_user(user_text)
        return self._drive()

    def _drive(self) -> ManagerTurn:
        for step in range(1, self.max_steps + 1):
            try:
                response = self.client.chat(self.messages)
            except ModelClientError as exc:
                return ManagerTurn("error", f"Model request failed: {exc}")

            LOGGER.info(
                "manager step %d response in %.3f s: prompt=%s cached=%s new=%s prefill=%s completion=%s generate=%s content=%r",
                step,
                response.elapsed_seconds,
                response.prompt_tokens if response.prompt_tokens is not None else "?",
                response.cached_tokens if response.cached_tokens is not None else "?",
                response.prompt_evaluated_tokens
                if response.prompt_evaluated_tokens is not None
                else "?",
                f"{response.prompt_seconds:.3f}s"
                if response.prompt_seconds is not None
                else "?",
                response.completion_tokens if response.completion_tokens is not None else "?",
                f"{response.generation_seconds:.3f}s"
                if response.generation_seconds is not None
                else "?",
                " ".join(response.content.strip().split())[:300],
            )
            self.messages.append({"role": "assistant", "content": response.content})
            directive = parse_manager_output(response.content)
            if directive.error:
                self._event(f"PROTOCOL_ERROR\n{directive.error}")
                continue

            if directive.action is ManagerAction.REPLY:
                return ManagerTurn("reply", directive.body)

            if directive.action is ManagerAction.ASK:
                return ManagerTurn("ask", directive.body)

            if directive.action is ManagerAction.WAIT:
                return ManagerTurn("wait", "Manager is waiting for an external event.")

            if directive.action is ManagerAction.DELEGATE:
                self._delegate(directive.skills, directive.body)
                continue

            if directive.action is ManagerAction.CONTINUE:
                assert directive.agent_id is not None
                self._continue_agent(directive.agent_id, directive.body)
                continue

        return ManagerTurn("error", f"Manager exceeded maximum of {self.max_steps} steps")

    def _delegate(self, skill_names: tuple[str, ...], task: str) -> None:
        try:
            skills = self.skill_base.require(skill_names)
        except SkillBaseError as exc:
            self._event(f"EVENT DELEGATE_FAILED\n{exc}")
            return

        worker = self.pool.acquire()
        if worker is None:
            self._event("EVENT DELEGATE_FAILED\nNo FREE agent container is available.")
            return

        self._event(
            f"EVENT STARTED {worker.agent_id}\nskills: {','.join(skill_names)}"
        )
        outcome = worker.start(task, skills)
        self._agent_outcome(outcome.agent_id, outcome.status, outcome.text)

    def _continue_agent(self, agent_id: str, context: str) -> None:
        worker = self.pool.get(agent_id)
        if worker is None:
            self._event(f"EVENT CONTINUE_FAILED {agent_id}\nUnknown agent id.")
            return
        if worker.state is not AgentState.WAITING:
            self._event(
                f"EVENT CONTINUE_FAILED {agent_id}\nAgent is {worker.state.value}, not WAITING."
            )
            return
        outcome = worker.continue_with(context)
        self._agent_outcome(outcome.agent_id, outcome.status, outcome.text)

    def _agent_outcome(self, agent_id: str, status: str, text: str) -> None:
        if status == "NEED":
            self._event(f"EVENT NEED {agent_id}\n{text}")
        else:
            self._event(f"EVENT RESULT {agent_id} {status}\n{text}")

    def _event(self, text: str) -> None:
        self.prompt_store.write_manager_prompt(text)
        self._append_user(text)

    def _append_user(self, text: str) -> None:
        content = text.strip()
        if self.messages and self.messages[-1]["role"] == "user":
            previous = self.messages[-1]["content"].rstrip()
            self.messages[-1]["content"] = f"{previous}\n\n{content}" if previous else content
            return
        self.messages.append({"role": "user", "content": content})

    def _bootstrap_prompt(self) -> str:
        return (
            "[AVAILABLE_SKILLS]\n"
            f"{self.skill_base.catalog_text()}\n"
            "[/AVAILABLE_SKILLS]\n\n"
            "[AGENT_CONTAINERS]\n"
            f"{self.pool.status_text()}\n"
            "[/AGENT_CONTAINERS]\n\n"
            "Система готова. Жди сообщения пользователя."
        )
