from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from .command_runtime import CommandRuntime
from .model_client import ModelClientError, OpenAIChatClient
from .prompt_store import AGENT_BOOTSTRAP_ACK, PromptStore
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
        self._steps_used = 0

    def start(self, task: str, skills: tuple[Skill, ...]) -> AgentOutcome:
        if self.state is not AgentState.FREE:
            raise RuntimeError(f"{self.agent_id} is {self.state}")
        self._skills = skills

        bootstrap = self.prompt_store.build_agent_bootstrap(skills, self.workspace)
        task_prompt = self.prompt_store.build_agent_task(task)
        self.prompt_store.write_agent_prompt(
            self.agent_id,
            bootstrap.rstrip() + "\n\n" + task_prompt,
        )

        reused = self._messages is not None and self._session_bootstrap == bootstrap
        if reused:
            assert self._messages is not None
            self._messages.append({"role": "user", "content": task_prompt})
        else:
            self._messages = [
                {
                    "role": "system",
                    "content": self.prompt_store.agent_system_prompt(self.agent_id),
                },
                {"role": "user", "content": bootstrap},
                {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
                {"role": "user", "content": task_prompt},
            ]
            self._session_bootstrap = bootstrap

        self._runtime = CommandRuntime(
            self.workspace,
            tuple(skill.name for skill in skills),
            max_file_bytes=self.max_file_bytes,
            timeout_seconds=self.command_timeout_seconds,
        )
        self._steps_used = 0
        self.state = AgentState.RUNNING
        LOGGER.info(
            "%s START skills=%s bootstrap_reused=%s workspace=%s task=%r",
            self.agent_id,
            ",".join(skill.name for skill in skills),
            reused,
            self.workspace,
            task,
        )
        return self._drive()

    def continue_with(self, context: str) -> AgentOutcome:
        if self.state is not AgentState.WAITING or self._messages is None:
            raise RuntimeError(f"{self.agent_id} is not waiting for context")
        self._messages.append({"role": "user", "content": f"ADDITIONAL CONTEXT:\n{context.strip()}"})
        self.state = AgentState.RUNNING
        LOGGER.info("%s CONTINUE context=%r", self.agent_id, context)
        return self._drive()

    def _drive(self) -> AgentOutcome:
        assert self._messages is not None
        assert self._runtime is not None
        repeated: dict[tuple[str, int, str, str], int] = {}

        while self._steps_used < self.max_steps:
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
                    {"role": "user", "content": self._runtime.format_protocol_error(directive.error)}
                )
                continue

            if directive.action is AgentAction.DONE:
                text = directive.body
                LOGGER.info("%s DONE steps=%d result=%r", self.agent_id, step, text)
                self._release(preserve_session=True)
                return AgentOutcome(self.agent_id, "OK", text, step)

            if directive.action is AgentAction.NEED:
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
            repeated[signature] = repeated.get(signature, 0) + 1
            if repeated[signature] > 2:
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

        used = self._steps_used
        LOGGER.error("%s FAILED exceeded maximum of %d model steps", self.agent_id, self.max_steps)
        self._release(preserve_session=False)
        return AgentOutcome(
            self.agent_id,
            "FAILED",
            f"agent exceeded maximum of {self.max_steps} model steps",
            used,
        )

    def _release(self, *, preserve_session: bool) -> None:
        self.state = AgentState.FREE
        if not preserve_session:
            self._messages = None
            self._session_bootstrap = None
        self._runtime = None
        self._skills = ()
        self._steps_used = 0
