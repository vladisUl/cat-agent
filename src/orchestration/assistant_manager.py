from __future__ import annotations

import logging
import math
import shlex
import time

from .agent import AgentState
from .event_store import EventStore, EventStoreError
from .manager import ManagerRuntime, ManagerTurn
from .model_client import ModelClientError
from .protocol import ManagerAction, parse_manager_output
from .skills import SkillBaseError
from .system_events import SystemEvent
from .tasks import TaskStoreError
from .workspace_command_runtime import unwrap_work_command

LOGGER = logging.getLogger(__name__)


class AssistantManagerRuntime(ManagerRuntime):
    """Single manager session: user dialogue, direct tools and task creation."""

    def __init__(self, *args, event_store: EventStore | None = None, **kwargs) -> None:
        self.event_store = event_store or EventStore()
        super().__init__(*args, **kwargs)

    def _bootstrap_prompt(self) -> str:
        # sys_prompt_manager.txt is the complete manager BASE.
        return ""

    def user_message(self, text: str) -> ManagerTurn:
        user_text = text.strip()
        folded = user_text.casefold()

        if folded == "чат":
            self._chat_mode = True
            self._close_chat_after_reply = False
        elif folded == "конец чата":
            self._close_chat_after_reply = True

        LOGGER.info(
            "MANAGER USER MESSAGE chat=%s raw=%r",
            self._chat_mode,
            user_text,
        )
        self.prompt_store.write_manager_prompt(f"[USER]\n{user_text}\n[/USER]")
        self._append_user(user_text)
        return self._drive_manager()

    def external_event(self, source: str, name: str) -> SystemEvent | None:
        """Resolve one real external event into the saved TASK/QUERY it activates."""
        source = source.strip().lower()
        name = name.strip()
        binding = self.event_store.resolve(source, name)
        if binding is None:
            LOGGER.warning(
                "SYSTEM external event ignored: no binding source=%s name=%s",
                source,
                name,
            )
            return None

        store = self.system_runtime.task_store
        if store is None:
            LOGGER.error("SYSTEM external event ignored: task store is not configured")
            return None
        task = store.get(binding.task_id)
        if task is None:
            LOGGER.warning(
                "SYSTEM external event ignored: binding=%s task=%d is missing",
                binding.name,
                binding.task_id,
            )
            return None
        if not task.enabled:
            LOGGER.info(
                "SYSTEM external event ignored: binding=%s task=%d is disabled",
                binding.name,
                binding.task_id,
            )
            return None

        LOGGER.info(
            "SYSTEM external event accepted source=%s name=%s task=%d description=%r",
            source,
            name,
            task.task_id,
            binding.description,
        )
        return SystemEvent(
            source=source,
            name=name,
            task="",
            created_monotonic=time.monotonic(),
            task_id=task.task_id,
        )

    def _drive_manager(self) -> ManagerTurn:
        for step in range(1, self.max_steps + 1):
            if not self.messages or self.messages[-1]["role"] != "user":
                self._abort_context()
                return ManagerTurn(
                    "error",
                    "TT violation: manager model call requires a fresh input",
                )

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
            LOGGER.info("manager step %d MODEL RESPONSE\n%s", step, response.content)
            self.messages.append({"role": "assistant", "content": response.content})

            output = response.content.strip()
            try:
                command = unwrap_work_command(output)
            except ValueError as exc:
                self._append_user(self._direct_runtime.format_protocol_error(str(exc)))
                continue

            if command is not None:
                result = self._execute_work_command(command)
                LOGGER.info("MANAGER WORK RESULT\n%s", result)
                self._append_user(result)
                continue

            directive = parse_manager_output(response.content)
            if directive.error:
                self._append_user(
                    self._direct_runtime.format_protocol_error(directive.error)
                )
                continue

            if directive.action is ManagerAction.ASK:
                LOGGER.info("MANAGER ASK %r", directive.body)
                return ManagerTurn("ask", directive.body)

            if directive.action is ManagerAction.REPLY:
                text = directive.body
                LOGGER.info("MANAGER REPLY %r", text)
                if self._close_chat_after_reply:
                    self._chat_mode = False
                    self._close_chat_after_reply = False
                    self._reset_to_base()
                elif not self._chat_mode:
                    self._reset_to_base()
                else:
                    LOGGER.info("MANAGER CHAT context preserved after REPLY")
                return ManagerTurn("reply", text)

            self._append_user(
                self._direct_runtime.format_protocol_error(
                    "manager response must be ASK, REPLY or /work#<command>"
                )
            )

        LOGGER.error("Manager exceeded maximum of %d steps", self.max_steps)
        self._abort_context()
        return ManagerTurn(
            "error",
            f"Manager exceeded maximum of {self.max_steps} steps",
        )

    def _execute_work_command(self, command: str) -> str:
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            return f"SYSTEM_ERROR\ninvalid command syntax: {exc}"
        if not argv:
            return "SYSTEM_ERROR\nempty command"

        if argv[0] in {"task_timer.sh", "query_timer.sh"}:
            return self._execute_task_command(argv)
        if argv[0] == "timer.sh":
            return self._execute_timer_command(argv)

        result = self._direct_runtime.execute(command)
        return self._direct_runtime.format_result(result)

    def _execute_task_command(self, argv: list[str]) -> str:
        if len(argv) != 5:
            return (
                "SYSTEM_ERROR\nusage: task_timer.sh|query_timer.sh "
                "PERIOD SKILLS DESCRIPTION TEXT"
            )

        try:
            period = float(argv[1])
        except ValueError:
            return "SYSTEM_ERROR\nperiod_seconds must be -1, 0 or > 0"
        if not math.isfinite(period) or period < -1 or (-1 < period < 0):
            return "SYSTEM_ERROR\nperiod_seconds must be -1, 0 or > 0"

        skill_names = tuple(
            item.strip() for item in argv[2].split(",") if item.strip()
        )
        if not skill_names or len(set(skill_names)) != len(skill_names):
            return "SYSTEM_ERROR\ninvalid skill list"

        description = argv[3].strip()
        task_text = argv[4].strip()
        if not description or not task_text:
            return "SYSTEM_ERROR\ndescription and task text must be non-empty"

        try:
            skills = self.skill_base.require(skill_names)
        except SkillBaseError as exc:
            return f"SYSTEM_ERROR\n{exc}"

        method = "task" if argv[0] == "task_timer.sh" else "query"
        if period == 0:
            return self._run_one_shot_agent(method, task_text, skills)
        if period == -1:
            return self._create_external_task(
                method,
                description,
                task_text,
                skill_names,
            )

        try:
            task = self.system_runtime.create_periodic_task(
                description,
                task_text,
                skill_names,
                period,
                method=method,
            )
        except (TaskStoreError, ValueError) as exc:
            return f"SYSTEM_ERROR\n{exc}"

        return (
            f"SYSTEM_OK\nTASK {task.task_id} created and started; "
            f"method={task.method} period={period:g}s "
            f"description={task.description}"
        )

    def _create_external_task(
        self,
        method: str,
        description: str,
        task_text: str,
        skill_names: tuple[str, ...],
    ) -> str:
        store = self.system_runtime.task_store
        if store is None:
            return "SYSTEM_ERROR\ntask store is not configured"

        try:
            task = store.create(
                description,
                task_text,
                method=method,
                skills=skill_names,
                timer_period_seconds=None,
                enabled=True,
            )
            try:
                binding = self.event_store.register(
                    task.task_id,
                    description,
                    source="gpio",
                )
            except Exception:
                store.delete(task.task_id)
                raise
        except (TaskStoreError, EventStoreError, OSError) as exc:
            return f"SYSTEM_ERROR\n{exc}"

        LOGGER.info(
            "SYSTEM external task created id=%d method=%s event=%s description=%r",
            task.task_id,
            task.method,
            binding.name,
            task.description,
        )
        return (
            f"SYSTEM_OK\nTASK {task.task_id} created for external event {binding.name}; "
            f"method={task.method} description={task.description}"
        )

    def _run_one_shot_agent(self, method: str, task_text: str, skills) -> str:
        worker = self.pool.acquire()
        if worker is None:
            return "SYSTEM_ERROR\nнет свободного агента"

        try:
            outcome = worker.start(task_text, skills, method=method)
        except Exception as exc:
            LOGGER.exception("one-shot agent failed")
            if worker.state is not AgentState.FREE:
                worker.sleep_to_base()
            return f"SYSTEM_ERROR\n{exc}"

        if outcome.status == "NEED":
            worker.sleep_to_base()
            return f"SYSTEM_ERROR\nагенту не хватает данных: {outcome.text}"
        if outcome.status != "OK":
            return f"SYSTEM_ERROR\n{outcome.text or outcome.status}"
        if method == "query":
            if not outcome.text.strip():
                return "SYSTEM_ERROR\nагент не вернул результат"
            return outcome.text.strip()
        return "SYSTEM_OK\nЗАДАНИЕ выполнено"

    def _execute_timer_command(self, argv: list[str]) -> str:
        if len(argv) < 2:
            return "SYSTEM_ERROR\nusage: timer.sh start|stop|period|delete|list ..."

        op = argv[1].lower()
        try:
            if op == "list" and len(argv) == 2:
                return f"SYSTEM_OK\n{self.system_runtime.task_status_text()}"

            if op in {"start", "stop", "delete"} and len(argv) == 3:
                task_id = int(argv[2])
                if task_id <= 0:
                    raise ValueError("task_number must be > 0")
                if op == "start":
                    self.system_runtime.start_task(task_id)
                    return f"SYSTEM_OK\nTASK {task_id} started"
                if op == "stop":
                    self.system_runtime.stop_task(task_id)
                    return f"SYSTEM_OK\nTASK {task_id} stopped"
                if not self.system_runtime.delete_task(task_id):
                    return f"SYSTEM_ERROR\nunknown task: {task_id}"
                self.event_store.unregister_task(task_id)
                return f"SYSTEM_OK\nTASK {task_id} deleted"

            if op == "period" and len(argv) == 4:
                period = float(argv[2])
                task_id = int(argv[3])
                if period <= 0 or task_id <= 0:
                    raise ValueError("period_seconds and task_number must be > 0")
                self.system_runtime.set_task_period(task_id, period)
                return f"SYSTEM_OK\nTASK {task_id} period changed to {period:g}s"
        except (ValueError, TaskStoreError, EventStoreError, OSError) as exc:
            return f"SYSTEM_ERROR\n{exc}"

        return "SYSTEM_ERROR\ninvalid timer.sh syntax"
