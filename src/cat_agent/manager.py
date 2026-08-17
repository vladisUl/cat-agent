from __future__ import annotations

from dataclasses import dataclass
import logging

from .agent import AgentState, AgentWorker
from .model_client import ModelClientError, OpenAIChatClient
from .pool import AgentPool
from .prompt_store import PromptStore
from .protocol import ManagerAction, ManagerDirective, parse_manager_output
from .skills import SkillBase, SkillBaseError
from .system_events import SystemEvent, SystemRuntime, TaskActivation
from .tasks import TaskRecord, TaskStoreError
from .workspace_command_runtime import CommandRuntime

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ManagerTurn:
    kind: str
    text: str


@dataclass(slots=True)
class AutonomousTaskExecution:
    activation: TaskActivation
    worker: AgentWorker


@dataclass(frozen=True, slots=True)
class AutonomousTaskCompletion:
    turn: ManagerTurn | None = None
    query_task_id: int | None = None
    query_result: str = ""


class ManagerRuntime:
    def __init__(
        self,
        client: OpenAIChatClient,
        skill_base: SkillBase,
        prompt_store: PromptStore,
        pool: AgentPool,
        system_runtime: SystemRuntime | None = None,
        *,
        max_steps: int,
        forced_delegate_skills: tuple[str, ...] | None = None,
    ) -> None:
        self.client = client
        self.skill_base = skill_base
        self.prompt_store = prompt_store
        self.pool = pool
        self.system_runtime = system_runtime or SystemRuntime()
        self.max_steps = max_steps
        self.forced_delegate_skills = forced_delegate_skills
        self._chat_mode = False
        self._close_chat_after_reply = False
        self._force_self = False

        template = self.pool.get("agent1")
        if template is None:
            raise RuntimeError("manager requires agent1 runtime template")
        self._manager_skills = self.skill_base.require(self.skill_base.names())
        self._manager_workspace = template.workspace
        self._self_runtime = CommandRuntime(
            template.workspace,
            tuple(skill.name for skill in self._manager_skills),
            max_file_bytes=template.max_file_bytes,
            timeout_seconds=template.command_timeout_seconds,
        )
        self._manager_tools_bootstrap = self.prompt_store.build_agent_bootstrap(
            self._manager_skills,
            self._manager_workspace,
        ).strip()

        self.system_runtime.set_task_handler(self._run_task_activation)
        bootstrap = self._bootstrap_prompt()
        system_context = (
            self.prompt_store.manager_system_prompt().strip()
            + "\n\n"
            + bootstrap.strip()
        )
        self.prompt_store.write_manager_prompt(system_context)
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": system_context},
        ]
        self._base_messages = [dict(item) for item in self.messages]

    def user_message(self, text: str) -> ManagerTurn:
        user_text = text.strip()
        folded = user_text.casefold()
        if folded == "чат":
            self._chat_mode = True
            self._close_chat_after_reply = False
        elif folded == "конец чата":
            self._close_chat_after_reply = True
            self._force_self = False
        elif folded == "сам" or folded.startswith("сам "):
            self._force_self = True

        LOGGER.info(
            "MANAGER USER MESSAGE chat=%s self=%s\n%s",
            self._chat_mode,
            self._force_self,
            user_text,
        )
        self.prompt_store.write_manager_prompt(f"[USER]\n{user_text}\n[/USER]")
        self._append_user(user_text)
        return self._drive()

    def system_event(self, event: SystemEvent) -> ManagerTurn:
        if event.task_id is not None:
            LOGGER.info(
                "SYSTEM autonomous event source=%s task_id=%d",
                event.source,
                event.task_id,
            )
            result = self.system_runtime.activate_task(
                event.task_id,
                source=event.source,
                name=event.name,
                now=event.created_monotonic,
            )
            if result is None:
                return ManagerTurn("silent", "")
            return self.autonomous_query_result(event.task_id, result)

        text = event.manager_text()
        LOGGER.info(
            "MANAGER SYSTEM EVENT source=%s name=%s\n%s",
            event.source,
            event.name,
            text,
        )
        self.prompt_store.write_manager_prompt(text)
        self._append_user(text)
        return self._drive()

    def autonomous_query_result(self, task_id: int, result: str) -> ManagerTurn:
        tick = f"SYSTEM_QUERY_RESULT TASK {task_id}\n{result.strip()}"
        LOGGER.info(
            "MANAGER autonomous QUERY tick task_id=%d\n%s",
            task_id,
            tick,
        )
        self._event(tick)
        return self._drive()

    def begin_autonomous_task(
        self,
        event: SystemEvent,
    ) -> AutonomousTaskExecution | AutonomousTaskCompletion:
        """Start a saved TASK/QUERY without consuming its first model TT."""
        if event.task_id is None:
            raise ValueError("autonomous task event requires task_id")
        store = self.system_runtime.task_store
        if store is None:
            return AutonomousTaskCompletion(turn=ManagerTurn("error", "Task store is not configured"))
        try:
            task = store.require(event.task_id)
        except TaskStoreError as exc:
            LOGGER.warning("SYSTEM TASK %d cannot start: %s", event.task_id, exc)
            return AutonomousTaskCompletion(turn=ManagerTurn("silent", ""))

        activation = TaskActivation(
            source=event.source.strip() or "system",
            name=event.name.strip(),
            task=task,
            created_monotonic=event.created_monotonic,
        )
        LOGGER.info(
            "SYSTEM task activation id=%d method=%s source=%s name=%s description=%r",
            task.task_id,
            task.method,
            activation.source,
            activation.name,
            task.description,
        )

        if not task.skills:
            LOGGER.error("SYSTEM TASK %d cannot run: no saved skills", task.task_id)
            return self._autonomous_error_completion(task, "нет сохранённых skills")

        try:
            skills = self.skill_base.require(task.skills)
        except SkillBaseError as exc:
            LOGGER.error(
                "SYSTEM TASK %d cannot run skills=%s error=%s",
                task.task_id,
                ",".join(task.skills),
                exc,
            )
            return self._autonomous_error_completion(task, str(exc))

        worker = self.pool.acquire()
        if worker is None:
            LOGGER.warning("SYSTEM TASK %d cannot start: no FREE agent", task.task_id)
            return self._autonomous_error_completion(task, "нет свободного агента")

        LOGGER.info(
            "SYSTEM TASK %d wake %s method=%s skills=%s description=%r",
            task.task_id,
            worker.agent_id,
            task.method,
            ",".join(task.skills),
            task.description,
        )
        try:
            worker.begin(task.text, skills, method=task.method)
        except Exception as exc:
            LOGGER.exception(
                "SYSTEM TASK %d agent %s failed during activation setup",
                task.task_id,
                worker.agent_id,
            )
            if worker.state is not AgentState.FREE:
                worker.sleep_to_base()
            return self._autonomous_error_completion(task, str(exc))

        return AutonomousTaskExecution(activation=activation, worker=worker)

    def step_autonomous_task(
        self,
        execution: AutonomousTaskExecution,
    ) -> AutonomousTaskCompletion | None:
        """Run exactly one agent TT. None means this activation resumes later."""
        task = execution.activation.task
        worker = execution.worker
        try:
            outcome = worker.step()
        except Exception as exc:
            LOGGER.exception(
                "SYSTEM TASK %d agent %s failed during TT step",
                task.task_id,
                worker.agent_id,
            )
            if worker.state is not AgentState.FREE:
                worker.sleep_to_base()
            return self._autonomous_error_completion(task, str(exc))

        if outcome is None:
            LOGGER.info(
                "SYSTEM TASK %d paused at TT boundary agent=%s",
                task.task_id,
                worker.agent_id,
            )
            return None

        LOGGER.info(
            "SYSTEM TASK %d outcome agent=%s status=%s steps=%d text=%r",
            task.task_id,
            outcome.agent_id,
            outcome.status,
            outcome.steps,
            outcome.text,
        )
        if outcome.status == "NEED":
            LOGGER.warning(
                "SYSTEM TASK %d autonomous agent returned NEED; dropping working context and sleeping to BASE",
                task.task_id,
            )
            worker.sleep_to_base()
            return self._autonomous_error_completion(task, outcome.text)

        if task.method == "task":
            return AutonomousTaskCompletion(turn=ManagerTurn("silent", ""))
        if outcome.status != "OK":
            return self._autonomous_error_completion(task, outcome.text or outcome.status)
        if not outcome.text.strip():
            return self._autonomous_error_completion(task, "агент не вернул значение")
        return AutonomousTaskCompletion(
            query_task_id=task.task_id,
            query_result=outcome.text,
        )

    def _autonomous_error_completion(
        self,
        task: TaskRecord,
        message: str,
    ) -> AutonomousTaskCompletion:
        if task.method != "query":
            return AutonomousTaskCompletion(turn=ManagerTurn("silent", ""))
        return AutonomousTaskCompletion(
            query_task_id=task.task_id,
            query_result=f"Ошибка запроса TASK {task.task_id}: {message}",
        )

    def _run_task_activation(self, activation: TaskActivation) -> str | None:
        task = activation.task

        def query_error(message: str) -> str | None:
            if task.method != "query":
                return None
            return f"Ошибка запроса TASK {task.task_id}: {message}"

        if not task.skills:
            LOGGER.error("SYSTEM TASK %d cannot run: no saved skills", task.task_id)
            return query_error("нет сохранённых skills")

        try:
            skills = self.skill_base.require(task.skills)
        except SkillBaseError as exc:
            LOGGER.error(
                "SYSTEM TASK %d cannot run skills=%s error=%s",
                task.task_id,
                ",".join(task.skills),
                exc,
            )
            return query_error(str(exc))

        worker = self.pool.acquire()
        if worker is None:
            LOGGER.warning("SYSTEM TASK %d skipped: no FREE agent", task.task_id)
            return query_error("нет свободного агента")

        LOGGER.info(
            "SYSTEM TASK %d wake %s method=%s skills=%s description=%r",
            task.task_id,
            worker.agent_id,
            task.method,
            ",".join(task.skills),
            task.description,
        )
        try:
            outcome = worker.start(task.text, skills, method=task.method)
        except Exception as exc:
            LOGGER.exception(
                "SYSTEM TASK %d agent %s failed during activation",
                task.task_id,
                worker.agent_id,
            )
            if worker.state is not AgentState.FREE:
                worker.sleep_to_base()
            return query_error(str(exc))

        LOGGER.info(
            "SYSTEM TASK %d outcome agent=%s status=%s steps=%d text=%r",
            task.task_id,
            outcome.agent_id,
            outcome.status,
            outcome.steps,
            outcome.text,
        )
        if outcome.status == "NEED":
            LOGGER.warning(
                "SYSTEM TASK %d autonomous agent returned NEED; dropping working context and sleeping to BASE",
                task.task_id,
            )
            worker.sleep_to_base()
            return query_error(outcome.text)

        if task.method == "task":
            return None
        if outcome.status != "OK":
            return query_error(outcome.text or outcome.status)
        if not outcome.text.strip():
            return query_error("агент не вернул значение")
        return outcome.text

    def _drive(self) -> ManagerTurn:
        last_protocol_signature: tuple[str, str] | None = None
        repeated_protocol_errors = 0

        for step in range(1, self.max_steps + 1):
            try:
                response = self.client.chat(self.messages)
            except ModelClientError as exc:
                LOGGER.exception("manager step %d model request failed", step)
                self._abort_context()
                return ManagerTurn("error", f"Model request failed: {exc}")

            LOGGER.info(
                "manager step %d response in %.3f s: prompt=%s cached=%s new=%s "
                "prefill=%s completion=%s generate=%s",
                step,
                response.elapsed_seconds,
                response.prompt_tokens if response.prompt_tokens is not None else "?",
                response.cached_tokens if response.cached_tokens is not None else "?",
                response.prompt_evaluated_tokens if response.prompt_evaluated_tokens is not None else "?",
                f"{response.prompt_seconds:.3f}s" if response.prompt_seconds is not None else "?",
                response.completion_tokens if response.completion_tokens is not None else "?",
                f"{response.generation_seconds:.3f}s" if response.generation_seconds is not None else "?",
            )
            LOGGER.info("manager step %d MODEL RESPONSE\n%s", step, response.content)
            self.messages.append({"role": "assistant", "content": response.content})
            directive = parse_manager_output(
                response.content,
                allow_command=self._force_self,
            )
            if directive.action is ManagerAction.DELEGATE and self._force_self:
                directive = ManagerDirective(
                    None,
                    "",
                    error="САМ forbids DELEGATE; execute one real tool command directly",
                )
            if directive.action is ManagerAction.SELF:
                directive = ManagerDirective(
                    None,
                    "",
                    error="SELF is obsolete; in САМ mode execute one real tool command directly",
                )

            if directive.error:
                LOGGER.warning("manager step %d protocol error: %s", step, directive.error)
                signature = (response.content.strip(), directive.error)
                if signature == last_protocol_signature:
                    repeated_protocol_errors += 1
                else:
                    last_protocol_signature = signature
                    repeated_protocol_errors = 1

                self._event(f"PROTOCOL_ERROR\n{directive.error}")
                if repeated_protocol_errors >= 3:
                    LOGGER.error(
                        "manager repeated identical protocol error %d times; aborting current request",
                        repeated_protocol_errors,
                    )
                    self._abort_context()
                    return ManagerTurn(
                        "error",
                        "Manager repeated the same invalid control response 3 times; current request aborted",
                    )
                continue

            last_protocol_signature = None
            repeated_protocol_errors = 0

            if directive.action is ManagerAction.REPLY:
                text = directive.body
                LOGGER.info("MANAGER REPLY %r", text)
                self._force_self = False
                if self._close_chat_after_reply:
                    self._chat_mode = False
                    self._close_chat_after_reply = False
                    self._reset_to_base()
                elif not self._chat_mode:
                    self._reset_to_base()
                else:
                    LOGGER.info("MANAGER CHAT context preserved after REPLY")
                return ManagerTurn("reply", text)

            if directive.action is ManagerAction.ASK:
                LOGGER.info("MANAGER ASK %r", directive.body)
                return ManagerTurn("ask", directive.body)

            if directive.action is ManagerAction.WAIT:
                LOGGER.info("MANAGER WAIT")
                return ManagerTurn("wait", "Manager is waiting for an external event.")

            if directive.action is ManagerAction.COMMAND:
                assert directive.command is not None
                LOGGER.info("MANAGER SELF TOOL COMMAND %s", directive.command)
                result = self._self_runtime.execute(directive.command)
                formatted = self._self_runtime.format_result(result)
                LOGGER.info(
                    "MANAGER SELF TOOL RESULT operation=%s exit=%d metadata=%r\n%s",
                    result.operation,
                    result.exit_code,
                    result.metadata,
                    formatted,
                )
                self._event(formatted)
                continue

            if directive.action is ManagerAction.SYSTEM:
                assert directive.system_command is not None
                LOGGER.info(
                    "MANAGER SYSTEM directive=%r body=%r",
                    directive.system_command,
                    directive.body,
                )
                if directive.system_command.startswith("TASK TIMER "):
                    result = self._execute_task_timer(directive)
                else:
                    result = self.system_runtime.execute(
                        directive.system_command,
                        directive.body,
                    )
                self._event(result)
                continue

            if directive.action is ManagerAction.DELEGATE:
                selected_skills = directive.skills
                if self.forced_delegate_skills is not None:
                    LOGGER.info(
                        "BENCH DELEGATE override model_skills=%s forced_skills=%s",
                        ",".join(directive.skills),
                        ",".join(self.forced_delegate_skills),
                    )
                    selected_skills = self.forced_delegate_skills
                LOGGER.info(
                    "MANAGER DELEGATE skills=%s task=%r",
                    ",".join(selected_skills),
                    directive.body,
                )
                self._delegate(selected_skills, directive.body)
                continue

            if directive.action is ManagerAction.CONTINUE:
                assert directive.agent_id is not None
                LOGGER.info(
                    "MANAGER CONTINUE agent=%s context=%r",
                    directive.agent_id,
                    directive.body,
                )
                self._continue_agent(directive.agent_id, directive.body)
                continue

        LOGGER.error("Manager exceeded maximum of %d steps", self.max_steps)
        self._abort_context()
        return ManagerTurn("error", f"Manager exceeded maximum of {self.max_steps} steps")

    def _execute_task_timer(self, directive: ManagerDirective) -> str:
        assert directive.system_command is not None
        parts = directive.system_command.split()
        try:
            op = parts[2]
            if op == "SET":
                if len(parts) != 4:
                    raise ValueError("invalid TASK TIMER SET command")
                period = float(parts[3])
                if directive.task_description is None:
                    raise ValueError("persistent task description is missing")
                if directive.task_method not in {"task", "query"}:
                    raise ValueError("persistent task method is missing")
                if not directive.skills:
                    raise ValueError("persistent task skills are missing")
                self.skill_base.require(directive.skills)
                task = self.system_runtime.create_periodic_task(
                    directive.task_description,
                    directive.body,
                    directive.skills,
                    period,
                    method=directive.task_method,
                )
                return (
                    f"SYSTEM_OK\nTASK {task.task_id} created and started; "
                    f"method={task.method} period={period:g}s description={task.description}"
                )

            if op in {"START", "STOP", "DELETE"}:
                if len(parts) != 4:
                    raise ValueError(f"invalid TASK TIMER {op} command")
                task_id = int(parts[3])
                if op == "START":
                    self.system_runtime.start_task(task_id)
                    return f"SYSTEM_OK\nTASK {task_id} started"
                if op == "STOP":
                    self.system_runtime.stop_task(task_id)
                    return f"SYSTEM_OK\nTASK {task_id} stopped"
                if not self.system_runtime.delete_task(task_id):
                    return f"SYSTEM_ERROR\nunknown task: {task_id}"
                return f"SYSTEM_OK\nTASK {task_id} deleted"

            if op == "PERIOD":
                if len(parts) != 5:
                    raise ValueError("invalid TASK TIMER PERIOD command")
                task_id = int(parts[3])
                period = float(parts[4])
                self.system_runtime.set_task_period(task_id, period)
                return f"SYSTEM_OK\nTASK {task_id} period changed to {period:g}s"

            if op == "LIST":
                return f"SYSTEM_OK\n{self.system_runtime.task_status_text()}"

            return f"SYSTEM_ERROR\nunknown TASK TIMER operation: {op}"
        except (ValueError, TaskStoreError, SkillBaseError) as exc:
            LOGGER.warning("SYSTEM persistent task command failed: %s", exc)
            return f"SYSTEM_ERROR\n{exc}"

    def _delegate(self, skill_names: tuple[str, ...], task: str) -> None:
        try:
            skills = self.skill_base.require(skill_names)
        except SkillBaseError as exc:
            LOGGER.warning("DELEGATE_FAILED skills=%s error=%s", skill_names, exc)
            self._event(f"EVENT DELEGATE_FAILED\n{exc}")
            return

        worker = self.pool.acquire()
        if worker is None:
            LOGGER.warning("DELEGATE_FAILED no FREE agent")
            self._event("EVENT DELEGATE_FAILED\nNo FREE agent container is available.")
            return

        LOGGER.info("AGENT assigned id=%s skills=%s", worker.agent_id, ",".join(skill_names))
        self._event(f"EVENT STARTED {worker.agent_id}\nskills: {','.join(skill_names)}")
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
        LOGGER.info("AGENT outcome id=%s status=%s text=%r", agent_id, status, text)
        if status == "NEED":
            self._event(f"EVENT NEED {agent_id}\n{text}")
        else:
            self._event(f"EVENT RESULT {agent_id} {status}\n{text}")

    def _event(self, text: str) -> None:
        LOGGER.info("MANAGER runtime event\n%s", text)
        self.prompt_store.write_manager_prompt(text)
        self._append_user(text)

    def _append_user(self, text: str) -> None:
        content = text.strip()
        if self.messages and self.messages[-1]["role"] == "user":
            previous = self.messages[-1]["content"].rstrip()
            self.messages[-1]["content"] = f"{previous}\n\n{content}" if previous else content
            return
        self.messages.append({"role": "user", "content": content})

    def _abort_context(self) -> None:
        self._chat_mode = False
        self._close_chat_after_reply = False
        self._force_self = False
        self._reset_to_base()

    def _reset_to_base(self) -> None:
        try:
            reset_to_base = getattr(self.client, "reset_to_base", None)
            if callable(reset_to_base):
                reset_to_base(self._base_messages)
            LOGGER.info("MANAGER RESET resident session to base")
        except Exception:
            LOGGER.exception("MANAGER failed to reset resident session to base")
        self.messages = [dict(item) for item in self._base_messages]

    def _bootstrap_prompt(self) -> str:
        return (
            "[AVAILABLE_SKILLS]\n"
            f"{self.skill_base.catalog_text()}\n"
            "[/AVAILABLE_SKILLS]\n\n"
            "[MANAGER_TOOLS]\n"
            f"{self._manager_tools_bootstrap}\n"
            "[/MANAGER_TOOLS]\n\n"
            "[AGENT_CONTAINERS]\n"
            f"{self.pool.status_text()}\n"
            "[/AGENT_CONTAINERS]\n\n"
            "[SYSTEM]\n"
            f"{self.system_runtime.capabilities_text()}\n"
            "[/SYSTEM]\n\n"
            "Система готова. Жди сообщения пользователя или SYSTEM_EVENT."
        )
