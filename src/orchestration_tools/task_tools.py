"""Native MANAGER tools for durable periodic QUERY tasks in shared TASK SYSTEM.

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

    def __init__(self, rpc, agent_permissions: Mapping[str, tuple[str, ...]]):
        self.rpc = rpc
        self.agent_ids = frozenset(agent_permissions)
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
            and record.get("timer_period_seconds") is not None
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

    def list_queries(self) -> dict:
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        LOGGER.info("native-task-tool LIST periodic queries")
        try:
            records = self.rpc.call("system", "task_snapshot")
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("LIST", exc)
        entries = [
            {
                "task_id": r["task_id"],
                "description": r["description"][len(TASK_PREFIX):],
                "period_seconds": r["timer_period_seconds"],
                "enabled": r["enabled"],
                "skills": r["skills"],
                "agent_id": r.get("agent_id"),
            }
            for r in records if self._ours(r)
        ]
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
            stopped = self.rpc.call("system", "stop_task", task_id)
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("STOP", exc)
        LOGGER.info("native-task-tool STOPPED id=%s", task_id)
        return {"ok": True, "task_id": task_id, "enabled": stopped["enabled"]}


def create_task_tools(controller: NativeTaskController) -> list:
    """Management tools only; delegate_task owns both task schedules."""
    def list_periodic_queries() -> dict:
        """Получить список периодических QUERY, созданных LiteRT MANAGER."""
        return controller.list_queries()

    def stop_periodic_query(task_id: int) -> dict:
        """Остановить ранее созданный периодический QUERY по номеру задачи.

        Args:
            task_id (int): Номер TASK из delegate_task или list_periodic_queries.
        """
        return controller.stop(task_id)

    return [list_periodic_queries, stop_periodic_query]
