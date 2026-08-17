from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from .workspace_command_runtime import CommandRuntime
from .model_client import ModelClientError, OpenAIChatClient
from .prompt_store import PromptStore
from .protocol import AgentAction, parse_agent_output
from .skills import Skill

LOGGER = logging.getLogger(__name__)


class AgentState(str, Enum):
    FREE = "FREE"
    RUNNING = "RUNNING"
    WAITING = "WAITING"


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    agent_id: str
    status: str
    text: str
    steps: int


class AgentWorker:
    def __init__(
        self,
        agent_id: str,
        client: OpenAIChatClient,
        prompt_store: PromptStore,
        workspace: Path,
        *,
        max_steps: int,
        max_file_bytes: int,
        command_timeout_seconds: int,
    ) -> None:
        self.agent_id = agent_id
        self.client = client
        self.prompt_store = prompt_store
        self.workspace = workspace
        self.max_steps = max_steps
        self.max_file_bytes = max_file_bytes
        self.command_timeout_seconds = command_timeout_seconds
        self.state = AgentState.FREE
        self._messages: list[dict[str, str]] | None = None
        self._session_bootstrap: str | None = None
        self._runtime: CommandRuntime | None = None
        self._skills: tuple[Skill, ...] = ()
        self._method: str | None = None
        self._steps_used = 0
        self._repeated: dict[tuple[str, int, str, str], int] = {}

    def begin(
        self,
        task: str,
        skills: tuple[Skill, ...],
        *,
        method: str | None = None,
    ) -> None:
        """Start one logical activation without consuming its first model TT."""
        if self.state is not AgentState.FREE:
            raise RuntimeError(f"{self.agent_id} is {self.state}")
        if method not in {None, "task", "query"}:
            raise ValueError(f"invalid task method: {method!r}")
        self._skills = skills
        self._method = method

        system_context = self.prompt_store.build_agent_system_context(
            self.agent_id,
            skills,
            self.workspace,
        )
        task_prompt = self.prompt_store.build_agent_task(task, method)
        self.prompt_store.write_agent_prompt(
            self.agent_id,
            system_context.rstrip() + "\n\n" + task_prompt,
        )

        reused = self._messages is not None and self._session_bootstrap == system_context
        if reused:
            assert self._messages is not None
            self._messages.append({"role": "user", "content": task_prompt})
        else:
            self._messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": task_prompt},
            ]
            self._session_bootstrap = system_context

        self._runtime = CommandRuntime(
            self.workspace,
            tuple(skill.name for skill in skills),
            max_file_bytes=self.max_file_bytes,
            timeout_seconds=self.command_timeout_seconds,
        )
        self._steps_used = 0
        self._repeated = {}
        self.state = AgentState.RUNNING
        LOGGER.info(
            "%s START method=%s skills=%s bootstrap_reused=%s workspace=%s task=%r",
            self.agent_id,
            method or "ordinary",
            ",".join(skill.name for skill in skills),
            reused,
            self.workspace,
            task,
        )

    def start(
        self,
        task: str,
        skills: tuple[Skill, ...],
        *,
        method: str | None = None,
    ) -> AgentOutcome:
        self.begin(task, skills, method=method)
        return self._drive()

    def step(self) -> AgentOutcome | None:
        """Execute exactly one model TICK->TOCK and stop at the next TT boundary."""
        if self.state is not AgentState.RUNNING:
            raise RuntimeError(f"{self.agent_id} is not RUNNING")
        assert self._messages is not None
        assert self._runtime is not None

        if not self._messages or self._messages[-1]["role"] != "user":
            raise RuntimeError(
                f"TT violation: {self.agent_id} model call requires a fresh user tick"
            )

        self._steps_used += 1
        step = self._steps_used
        try:
            response = self.client.chat(self._messages)
        except ModelClientError as exc:
            LOGGER.exception("%s step %d model request failed", self.agent_id, step)
            self._release(preserve_session=False)
            return AgentOutcome(self.agent_id, "FAILED", str(exc), step)

        LOGGER.info(
            "%s step %d response in %.3f s: prompt=%s cached=%s new=%s "
            "prefill=%s completion=%s generate=%s",
            self.agent_id,
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
        )
        LOGGER.info("%s step %d MODEL RESPONSE\n%s", self.agent_id, step, response.content)
        self._messages.append({"role": "assistant", "content": response.content})
        directive = parse_agent_output(response.content)

        if directive.error:
            LOGGER.warning(
                "%s step %d protocol error: %s",
                self.agent_id,
                step,
                directive.error,
            )
            self._messages.append(
                {
                    "role": "user",
                    "content": self._runtime.format_protocol_error(directive.error),
                }
            )
            return self._continue_or_limit(step)

        if directive.action is AgentAction.DONE:
            if self._method == "query" and not directive.body:
                message = 'query completion requires a non-empty string in {"result":"..."}'
                LOGGER.warning("%s step %d protocol error: %s", self.agent_id, step, message)
                self._messages.append(
                    {"role": "user", "content": self._runtime.format_protocol_error(message)}
                )
                return self._continue_or_limit(step)
            if self._method == "task" and directive.body:
                message = 'task completion must be {"done":true}'
                LOGGER.warning("%s step %d protocol error: %s", self.agent_id, step, message)
                self._messages.append(
                    {"role": "user", "content": self._runtime.format_protocol_error(message)}
                )
                return self._continue_or_limit(step)
            if self._method is None and not directive.body:
                message = 'ordinary task completion requires a non-empty string in {"result":"..."}'
                LOGGER.warning("%s step %d protocol error: %s", self.agent_id, step, message)
                self._messages.append(
                    {"role": "user", "content": self._runtime.format_protocol_error(message)}
                )
                return self._continue_or_limit(step)

            text = "" if self._method == "task" else directive.body
            LOGGER.info(
                "%s COMPLETE method=%s steps=%d result=%r",
                self.agent_id,
                self._method or "ordinary",
                step,
                text,
            )
            self._release(preserve_session=True)
            return AgentOutcome(self.agent_id, "OK", text, step)

        if directive.action is AgentAction.NEED:
            if self._method == "query":
                message = 'query must return {"result":"..."}; {"need":"..."} is not allowed'
                LOGGER.warning("%s step %d protocol error: %s", self.agent_id, step, message)
                self._messages.append(
                    {"role": "user", "content": self._runtime.format_protocol_error(message)}
                )
                return self._continue_or_limit(step)
            self.state = AgentState.WAITING
            LOGGER.info("%s NEED steps=%d text=%r", self.agent_id, step, directive.body)
            return AgentOutcome(self.agent_id, "NEED", directive.body, step)

        assert directive.command is not None
        LOGGER.info("%s step %d TOOL COMMAND %s", self.agent_id, step, directive.command)
        result = self._runtime.execute(directive.command)
        LOGGER.info(
            "%s step %d TOOL RESULT operation=%s exit=%d metadata=%r\n%s",
            self.agent_id,
            step,
            result.operation,
            result.exit_code,
            result.metadata,
            self._runtime.format_result(result),
        )
        signature = (directive.command, result.exit_code, result.stdout, result.stderr)
        self._repeated[signature] = self._repeated.get(signature, 0) + 1
        if self._repeated[signature] > 2:
            LOGGER.error(
                "%s FAILED same command/result repeated more than twice: %s",
                self.agent_id,
                directive.command,
            )
            self._release(preserve_session=False)
            return AgentOutcome(
                self.agent_id,
                "FAILED",
                "same command with the same result repeated more than twice",
                step,
            )
        self._messages.append({"role": "user", "content": self._runtime.format_result(result)})
        return self._continue_or_limit(step)

    def continue_with(self, context: str) -> AgentOutcome:
        if self.state is not AgentState.WAITING or self._messages is None:
            raise RuntimeError(f"{self.agent_id} is not waiting for context")
        context_prompt = self.prompt_store.build_agent_context(context)
        self._messages.append({"role": "user", "content": context_prompt})
        self.state = AgentState.RUNNING
        LOGGER.info("%s CONTINUE context=%r", self.agent_id, context)
        return self._drive()

    def sleep_to_base(self) -> None:
        """Finish an autonomous activation and leave only the reusable BASE context."""
        LOGGER.info("%s SLEEP TO BASE state=%s", self.agent_id, self.state.value)
        self._release(preserve_session=True)

    def _drive(self) -> AgentOutcome:
        while True:
            outcome = self.step()
            if outcome is not None:
                return outcome

    def _continue_or_limit(self, step: int) -> AgentOutcome | None:
        if self._steps_used < self.max_steps:
            return None
        used = self._steps_used
        LOGGER.error("%s FAILED exceeded maximum of %d model steps", self.agent_id, self.max_steps)
        self._release(preserve_session=False)
        return AgentOutcome(
            self.agent_id,
            "FAILED",
            f"agent exceeded maximum of {self.max_steps} model steps",
            used if used else step,
        )

    def _release(self, *, preserve_session: bool) -> None:
        self.state = AgentState.FREE
        if preserve_session and self._messages is not None:
            base_messages = [dict(self._messages[0])]
            try:
                reset_to_base = getattr(self.client, "reset_to_base", None)
                if callable(reset_to_base):
                    reset_to_base(base_messages)
                self._messages = base_messages
                LOGGER.info("%s RESET resident session to base", self.agent_id)
            except Exception:
                LOGGER.exception("%s failed to reset resident session to base", self.agent_id)
                self._messages = None
                self._session_bootstrap = None
        else:
            self._messages = None
            self._session_bootstrap = None
        self._runtime = None
        self._skills = ()
        self._method = None
        self._steps_used = 0
        self._repeated = {}
