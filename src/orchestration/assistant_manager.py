from __future__ import annotations

import logging
import math
import shlex
import time

from .agent import AgentState
from .event_store import EventStore, EventStoreError
from .manager import (
    AutonomousTaskCompletion,
    AutonomousTaskExecution,
    ManagerRuntime,
    ManagerTurn,
)
from .model_client import ModelClientError
from .mqtt_events import MqttTopicCatalog
from .protocol import ManagerAction, parse_manager_output
from .skills import SkillBaseError
from .system_events import SystemEvent, TaskActivation
from .tasks import TaskStoreError
from .workspace_command_runtime import unwrap_work_command

LOGGER = logging.getLogger(__name__)


class AssistantManagerRuntime(ManagerRuntime):
    """Single manager session: user dialogue, direct tools and task creation."""

    def __init__(self, *args, event_store: EventStore | None = None, **kwargs) -> None:
        self.event_store = event_store or EventStore()
        super().__init__(*args, **kwargs)
        self.mqtt_catalog = MqttTopicCatalog(self.prompt_store.prompt_dir / "mqtt.txt")

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

    def human_session_released(self) -> None:
        """Drop transient dialogue state when its human client goes away."""
        if self._chat_mode:
            LOGGER.info("MANAGER human session released; CHAT context preserved")
            return
        self._abort_context()
        LOGGER.info("MANAGER human session released; context reset to BASE")

    def external_event(
        self,
        source: str,
        name: str,
        *,
        value: str | None = None,
    ) -> SystemEvent | None:
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
            task="" if value is None else value.strip(),
            created_monotonic=time.monotonic(),
            task_id=task.task_id,
        )

    def begin_autonomous_task(
        self,
        event: SystemEvent,
    ) -> AutonomousTaskExecution | AutonomousTaskCompletion:
        """Resume MQTT events after their deferred mqtt_sub tool call."""
        if (
            event.source.strip().lower() != "mqtt"
            or event.task_id is None
            or not event.task.strip()
        ):
            return super().begin_autonomous_task(event)

        store = self.system_runtime.task_store
        if store is None:
            return AutonomousTaskCompletion(
                turn=ManagerTurn("error", "Task store is not configured")
            )
        try:
            task = store.require(event.task_id)
        except TaskStoreError as exc:
            LOGGER.warning("SYSTEM TASK %d cannot start: %s", event.task_id, exc)
            return AutonomousTaskCompletion(turn=ManagerTurn("silent", ""))

        activation = TaskActivation(
            source="mqtt",
            name=event.name.strip(),
            task=task,
            created_monotonic=event.created_monotonic,
        )
        if not task.skills:
            return self._autonomous_error_completion(task, "нет сохранённых skills")
        try:
            skills = self.skill_base.require(task.skills)
        except SkillBaseError as exc:
            return self._autonomous_error_completion(task, str(exc))

        binding = self.event_store.resolve("mqtt", activation.name)
        if binding is None:
            return self._autonomous_error_completion(
                task,
                "MQTT binding не найден",
            )

        worker = self.pool.acquire_event()
        if worker is None:
            return self._autonomous_error_completion(
                task,
                "нет свободного событийного агента",
            )

        try:
            worker.begin_with_tool_result(
                task.text,
                skills,
                binding.command,
                event.task,
                method=task.method,
            )
        except Exception as exc:
            LOGGER.exception(
                "SYSTEM TASK %d MQTT resume failed agent=%s",
                task.task_id,
                worker.agent_id,
            )
            if worker.state is not AgentState.FREE:
                worker.sleep_to_base()
            return self._autonomous_error_completion(task, str(exc))

        return AutonomousTaskExecution(activation=activation, worker=worker)

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
                if result is None:
                    LOGGER.info("MANAGER WORK RESULT silent")
                    if not self._chat_mode:
                        self._reset_to_base()
                    else:
                        LOGGER.info("MANAGER CHAT context preserved after silent work result")
                    return ManagerTurn("silent", "")
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

    def _execute_work_command(self, command: str) -> str | None:
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            return f"SYSTEM_ERROR\ninvalid command syntax: {exc}"
        if not argv:
            return "SYSTEM_ERROR\nempty command"

        if argv[0] in {"task_timer.sh", "query_timer.sh"}:
            usage = (
                "SYSTEM_ERROR\nusage: task_timer.sh|query_timer.sh "
                "PERIOD SKILLS -- TEXT"
            )
            try:
                separator = argv.index("--", 2)
            except ValueError:
                return usage
            if separator < 3 or separator != len(argv) - 2:
                return usage
            normalized = [argv[0], argv[1], *argv[2:separator], argv[separator + 1]]
            return self._execute_task_command(normalized)
        if argv[0] == "timer.sh":
            return self._execute_timer_command(argv)

        result = self._direct_runtime.execute(command)
        return self._direct_runtime.format_result(result)

    def _execute_task_command(self, argv: list[str]) -> str | None:
        if len(argv) < 4:
            return (
                "SYSTEM_ERROR\nusage: task_timer.sh|query_timer.sh "
                "PERIOD SKILLS TEXT"
            )

        try:
            period = float(argv[1])
        except ValueError:
            return "SYSTEM_ERROR\nperiod_seconds must be -1, 0 or > 0"
        if not math.isfinite(period) or period < -1 or (-1 < period < 0):
            return "SYSTEM_ERROR\nperiod_seconds must be -1, 0 or > 0"

        skill_names = tuple(item.strip() for item in argv[2:-1] if item.strip())
        if not skill_names or len(set(skill_names)) != len(skill_names):
            return "SYSTEM_ERROR\ninvalid skill list"

        task_text = argv[-1].strip()
        if not task_text:
            return "SYSTEM_ERROR\ntask text must be non-empty"

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
                task_text,
                skill_names,
                skills,
            )

        try:
            task = self.system_runtime.create_periodic_task(
                task_text,
                task_text,
                skill_names,
                period,
                method=method,
            )
        except (TaskStoreError, ValueError) as exc:
            return f"SYSTEM_ERROR\n{exc}"

        return f"SYSTEM_OK\nTASK {task.task_id} created and started"

    def _create_external_task(
        self,
        method: str,
        task_text: str,
        skill_names: tuple[str, ...],
        skills,
    ) -> str:
        if "mqtt" not in skill_names:
            return "SYSTEM_ERROR\nexternal event currently requires mqtt skill"

        store = self.system_runtime.task_store
        if store is None:
            return "SYSTEM_ERROR\ntask store is not configured"

        worker = self.pool.acquire_event()
        if worker is None:
            return "SYSTEM_ERROR\nнет свободного событийного агента"

        try:
            command = worker.plan_first_command(task_text, skills, method=method)
        except Exception as exc:
            LOGGER.exception("external MQTT task planning failed")
            if worker.state is not AgentState.FREE:
                worker.sleep_to_base()
            return f"SYSTEM_ERROR\n{exc}"

        try:
            command_argv = shlex.split(command, posix=True)
        except ValueError as exc:
            return f"SYSTEM_ERROR\ninvalid agent mqtt command: {exc}"
        if len(command_argv) != 3 or command_argv[0] != "mqtt_sub.sh":
            return (
                "SYSTEM_ERROR\nexternal MQTT event requires agent command "
                "mqtt_sub.sh TOPIC FIELD"
            )
        topic, field = command_argv[1], command_argv[2]

        try:
            rule = self.mqtt_catalog.require(topic, field)
        except (FileNotFoundError, ValueError) as exc:
            return f"SYSTEM_ERROR\n{exc}"

        try:
            task = store.create(
                task_text,
                task_text,
                method=method,
                skills=skill_names,
                timer_period_seconds=None,
                enabled=True,
            )
            try:
                binding = self.event_store.register(
                    task.task_id,
                    task_text,
                    source="mqtt",
                    topic=topic,
                    field=field,
                    value_type=rule.value_type,
                    values=rule.values,
                    command=command,
                )
            except Exception:
                store.delete(task.task_id)
                raise
        except (TaskStoreError, EventStoreError, OSError) as exc:
            return f"SYSTEM_ERROR\n{exc}"

        LOGGER.info(
            "SYSTEM external MQTT task created id=%d method=%s event=%s topic=%s field=%s values=%s",
            task.task_id,
            task.method,
            binding.name,
            binding.topic,
            binding.field,
            ",".join(binding.values),
        )
        return f"SYSTEM_OK\nTASK {task.task_id} created and started"

    def _run_one_shot_agent(self, method: str, task_text: str, skills) -> str | None:
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
            result = outcome.text.strip()
            return result or None
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
