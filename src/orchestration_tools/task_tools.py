"""Native MANAGER tools for durable periodic and MQTT QUERY tasks in TASK SYSTEM.

A native tool may run synchronously inside send_message(); it must never
perform nested inference or create a local timer. Only authoritative RPC
mutates persistent task state. A lost RPC reply is an UNKNOWN outcome.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

from orchestration.tasks import TaskStoreError

LOGGER = logging.getLogger(__name__)
MAX_PERIOD_SECONDS = 604800
MAX_INSTRUCTION_CHARS = 4096
TASK_PREFIX = "LiteRT QUERY: "


def _period_seconds(value: object) -> int | None:
    """Accept exact whole seconds even if native tool decoding used float/str.

    Never round fractional intervals or accept bool as an integer. Keep
    this conversion at the native-tool boundary; RPC remains strictly typed.
    """
    if type(value) is int:
        seconds = value
    elif type(value) is float:
        if not 10 <= value <= MAX_PERIOD_SECONDS or not value.is_integer():
            return None
        seconds = int(value)
    elif type(value) is str:
        raw = value.strip()
        if not 1 <= len(raw) <= 16:
            return None
        whole, dot, fraction = raw.partition(".")
        if not whole.isascii() or not whole.isdecimal():
            return None
        if dot and (not fraction or any(char != "0" for char in fraction)):
            return None
        seconds = int(whole)
    else:
        return None
    return seconds if 10 <= seconds <= MAX_PERIOD_SECONDS else None


def _task_id(value: object) -> int | None:
    """Decode a positive TASK id exactly; native LiteRT may supply a string."""
    if type(value) is int:
        identifier = value
    elif type(value) is float:
        if not value.is_integer() or value < 1 or value > 10**15:
            return None
        identifier = int(value)
    elif type(value) is str:
        raw = value.strip()
        if not 1 <= len(raw) <= 16:
            return None
        whole, dot, fraction = raw.partition(".")
        if not whole.isascii() or not whole.isdecimal():
            return None
        if dot and (not fraction or any(char != "0" for char in fraction)):
            return None
        identifier = int(whole)
    else:
        return None
    return identifier if 1 <= identifier <= 10**15 else None


class NativeTaskController:
    """Restrict persistent mutations to an active human MANAGER turn."""

    def __init__(self, rpc, agent_permissions: Mapping[str, tuple[str, ...]],
                 event_tools: Mapping[str, dict] | None = None):
        self.rpc = rpc
        self.agent_permissions = {
            name: frozenset(names) for name, names in agent_permissions.items()
        }
        self.agent_ids = frozenset(agent_permissions)
        self.event_tools = dict(event_tools or {})
        self.enabled = False
        self.in_turn = False
        self._create_attempted = False

    def begin_turn(self) -> None:
        if self.in_turn:
            raise RuntimeError("Nested native TASK management turn")
        self.in_turn = True
        self._create_attempted = False

    def finish_turn(self) -> None:
        self.in_turn = False

    def _allowed(self) -> bool:
        return self.enabled and self.in_turn

    @staticmethod
    def _ours(record: dict) -> bool:
        return (
            isinstance(record, dict)
            and isinstance(record.get("description"), str)
            and record["description"].startswith(TASK_PREFIX)
            and record.get("method") == "query"
            and record.get("executor") == "litert"
        )

    @staticmethod
    def _rpc_error(operation: str, exc: Exception) -> dict:
        LOGGER.warning("native-task-tool %s RPC error: %s", operation, exc)
        # The service rolls back rejected transactions. A transport loss
        # is different: the service may have committed before reply loss.
        transport = (
            not isinstance(exc, TaskStoreError)
            or "Task SYSTEM unavailable" in str(exc)
        )
        return {
            "ok": False,
            "status": "unknown" if transport else "rejected",
            "error": (
                "TASK SYSTEM reply may have been lost; list saved tasks "
                "before retrying: " if transport else "TASK SYSTEM rejected request: "
            ) + str(exc),
        }

    def create(self, period_seconds: int, agent_id: str, instruction: str) -> dict:
        """Persist the exact same AGENT task text used for one-shot delegation."""
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        LOGGER.info(
            "native-task-tool CREATE ARG period_seconds=%r type=%s agent=%r",
            period_seconds, type(period_seconds).__name__, agent_id,
        )
        period = _period_seconds(period_seconds)
        if period is None:
            return {"ok": False, "error": "period_seconds must be an integer in 10..604800"}
        if not isinstance(agent_id, str) or agent_id not in self.agent_ids:
            return {"ok": False, "error": "unknown_agent"}
        if (not isinstance(instruction, str) or not instruction.strip()
                or len(instruction) > MAX_INSTRUCTION_CHARS):
            return {"ok": False, "error": "invalid_task"}
        try:
            instruction.encode("utf-8", "strict")
        except UnicodeEncodeError:
            return {"ok": False, "error": "invalid_unicode"}
        if self._create_attempted:
            return {"ok": False, "error": "Already attempted creating a task in this turn; list tasks before retrying"}
        self._create_attempted = True

        instruction = instruction.strip()
        LOGGER.info("native-task-tool CREATE period=%d agent=%s task=%r",
                    period, agent_id, instruction)
        try:
            record = self.rpc.call(
                "system", "create_periodic_task",
                TASK_PREFIX + instruction[:100], instruction, [], float(period),
                method="query", executor="litert", agent_id=agent_id,
            )
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("CREATE", exc)
        LOGGER.info("native-task-tool CREATED id=%s period=%s agent=%s",
                    record["task_id"], period, agent_id)
        return {
            "ok": True, "task_id": record["task_id"], "agent_id": agent_id,
            "period_seconds": period, "status": "scheduled",
            "delivery": "TASK QUERY notifications inbox",
        }

    def create_external(self, agent_id: str, instruction: str, mqtt_tool: str) -> dict:
        """Create a durable MQTT-triggered QUERY, with -1 as the public sentinel."""
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        if not isinstance(agent_id, str) or agent_id not in self.agent_ids:
            return {"ok": False, "error": "unknown_agent"}
        if (not isinstance(instruction, str) or not instruction.strip()
                or len(instruction) > MAX_INSTRUCTION_CHARS):
            return {"ok": False, "error": "invalid_task"}
        if not isinstance(mqtt_tool, str) or mqtt_tool not in self.event_tools:
            return {"ok": False, "error": "unknown_mqtt_event_tool"}
        if mqtt_tool not in self.agent_permissions[agent_id]:
            return {"ok": False, "error": "mqtt_tool_not_granted_to_agent"}
        spec = self.event_tools[mqtt_tool]
        # TASK SYSTEM currently has one broker subscription; reject signals
        # attached to another broker rather than saving an untriggerable TASK.
        from orchestration.mqtt_events import MQTT_HOST, MQTT_PORT
        if (spec["host"], spec["port"]) != (MQTT_HOST, MQTT_PORT):
            return {"ok": False, "error": "mqtt_event_broker_not_monitored"}
        if self._create_attempted:
            return {"ok": False, "error": "Already attempted creating a task in this turn; list tasks before retrying"}
        self._create_attempted = True
        instruction = instruction.strip()
        topic, field = spec["topic"], spec["field"]
        task_kwargs = dict(
            description=TASK_PREFIX + instruction[:100],
            text=instruction, method="query", skills=[mqtt_tool],
            timer_period_seconds=None, enabled=True, executor="litert",
            agent_id=agent_id,
        )
        binding_kwargs = dict(
            source="mqtt", topic=topic, field=field, value_type="boolean",
            values=["true", "false"], command=f"mqtt_sub.sh {topic} {field}",
        )
        try:
            record = self.rpc.call(
                "system", "create_external_task", task_kwargs, binding_kwargs,
            )
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("CREATE MQTT", exc)
        LOGGER.info("native-task-tool MQTT CREATED id=%s agent=%s topic=%s field=%s",
                    record["task_id"], agent_id, topic, field)
        return {
            "ok": True, "task_id": record["task_id"],
            "agent_id": agent_id, "period_seconds": -1,
            "mqtt_tool": mqtt_tool, "topic": topic, "field": field,
            "status": "scheduled", "delivery": "TASK QUERY notifications inbox",
        }

    def list_queries(self) -> dict:
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        LOGGER.info("native-task-tool LIST periodic queries")
        try:
            records = self.rpc.call("system", "task_snapshot")
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("LIST", exc)
        try:
            bindings = self.rpc.call("events", "snapshot")
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("LIST MQTT", exc)
        by_task = {
            binding["task_id"]: binding for binding in bindings
            if binding.get("source") == "mqtt"
        }
        entries = []
        for record in records:
            if not self._ours(record):
                continue
            external = record["timer_period_seconds"] is None
            binding = by_task.get(record["task_id"]) if external else None
            if external and binding is None:
                # A malformed legacy record must not appear as a live sensor.
                continue
            entry = {
                "task_id": record["task_id"],
                "description": record["description"][len(TASK_PREFIX):],
                "period_seconds": -1 if external else record["timer_period_seconds"],
                "enabled": record["enabled"],
                "skills": record["skills"],
                "agent_id": record.get("agent_id"),
            }
            if external:
                entry.update(topic=binding["topic"], field=binding["field"],
                             event_name=binding["name"])
            entries.append(entry)
        LOGGER.info("native-task-tool LIST found=%d", len(entries))
        return {"ok": True, "tasks": entries}

    def stop(self, task_id: int) -> dict:
        LOGGER.info("native-task-tool STOP ARG task_id=%r type=%s",
                    task_id, type(task_id).__name__)
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        normalized_id = _task_id(task_id)
        if normalized_id is None:
            LOGGER.warning("native-task-tool STOP REJECT invalid task_id=%r", task_id)
            return {"ok": False, "error": "task_id must be a positive integer"}
        task_id = normalized_id
        try:
            record = self.rpc.call("tasks", "get", task_id)
            if record is None or not self._ours(record):
                LOGGER.warning("native-task-tool STOP NOT-FOUND id=%s", task_id)
                return {"ok": False, "error": "native periodic QUERY not found"}
            stopped = self.rpc.call("system", "pause_task", task_id)
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("STOP", exc)
        LOGGER.info("native-task-tool STOPPED id=%s", task_id)
        return {"ok": True, "task_id": task_id, "enabled": stopped["enabled"]}

    def start(self, task_id: int) -> dict:
        """Resume a native periodic TASK with its persisted instruction and timer."""
        LOGGER.info("native-task-tool START ARG task_id=%r type=%s",
                    task_id, type(task_id).__name__)
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        normalized_id = _task_id(task_id)
        if normalized_id is None:
            LOGGER.warning("native-task-tool START REJECT invalid task_id=%r", task_id)
            return {"ok": False, "error": "task_id must be a positive integer"}
        try:
            record = self.rpc.call("tasks", "get", normalized_id)
            if record is None or not self._ours(record):
                LOGGER.warning("native-task-tool START NOT-FOUND id=%s", normalized_id)
                return {"ok": False, "error": "native periodic QUERY not found"}
            if record["enabled"]:
                # Do not unnecessarily change generation or reset an active timer.
                LOGGER.info("native-task-tool ALREADY RUNNING id=%s", normalized_id)
                return {"ok": True, "task_id": normalized_id,
                        "enabled": True, "already_running": True,
                        "period_seconds": record["timer_period_seconds"]}
            started = self.rpc.call("system", "resume_task", normalized_id)
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("START", exc)
        LOGGER.info("native-task-tool STARTED id=%s period=%s",
                    normalized_id, started["timer_period_seconds"])
        return {"ok": True, "task_id": normalized_id,
                "enabled": started["enabled"], "already_running": False,
                "period_seconds": started["timer_period_seconds"]}

    def delete(self, task_id: int) -> dict:
        """Permanently remove one native periodic TASK from the shared service."""
        LOGGER.info("native-task-tool DELETE ARG task_id=%r type=%s",
                    task_id, type(task_id).__name__)
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        normalized_id = _task_id(task_id)
        if normalized_id is None:
            LOGGER.warning("native-task-tool DELETE REJECT invalid task_id=%r", task_id)
            return {"ok": False, "error": "task_id must be a positive integer"}
        try:
            record = self.rpc.call("tasks", "get", normalized_id)
            if record is None or not self._ours(record):
                LOGGER.warning("native-task-tool DELETE NOT-FOUND id=%s", normalized_id)
                return {"ok": False, "error": "native periodic QUERY not found"}
            deleted = self.rpc.call("system", "delete_task", normalized_id)
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("DELETE", exc)
        if deleted is not True:
            LOGGER.warning("native-task-tool DELETE NOT-FOUND after lookup id=%s",
                           normalized_id)
            return {"ok": False, "error": "native periodic QUERY not found"}
        LOGGER.info("native-task-tool DELETED id=%s", normalized_id)
        return {"ok": True, "task_id": normalized_id, "deleted": True}


def create_task_tools(controller: NativeTaskController) -> list:
    """Management tools only; delegate_task owns one-shot and saved tasks."""
    def list_periodic_queries() -> dict:
        """Список сохранённых LiteRT QUERY: period_seconds=-1 означает MQTT-событие."""
        return controller.list_queries()

    def start_periodic_query(task_id: int) -> dict:
        """Возобновить сохранённый LiteRT QUERY (таймер или внешнее MQTT-событие).

        Для таймера первый запуск после одного полного периода.
        MQTT-подписка остаётся в TASK SYSTEM; текст задания не меняется.

        Args:
            task_id (int): Номер TASK из delegate_task или list_periodic_queries.
        """
        return controller.start(task_id)

    def stop_periodic_query(task_id: int) -> dict:
        """Остановить ранее созданный периодический или MQTT QUERY по номеру.

        Args:
            task_id (int): Номер TASK из delegate_task или list_periodic_queries.
        """
        return controller.stop(task_id)

    def delete_periodic_query(task_id: int) -> dict:
        """Удалить сохранённое периодическое или MQTT задание LiteRT.

        Остановка (stop_periodic_query) сохраняет TASK в списке.
        Удаление освобождает запись и таймер; повторный запуск потребует нового задания.

        Args:
            task_id (int): Номер TASK из delegate_task или list_periodic_queries.
        """
        return controller.delete(task_id)

    return [list_periodic_queries, start_periodic_query,
            stop_periodic_query, delete_periodic_query]
