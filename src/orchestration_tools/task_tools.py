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
MAX_INSTRUCTION_CHARS = 2048
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


class NativeTaskController:
    """Restrict persistent mutations to an active human MANAGER turn."""

    def __init__(self, rpc, agent_permissions: Mapping[str, tuple[str, ...]]):
        self.rpc = rpc
        self.allowed_skills = frozenset(
            name for names in agent_permissions.values() for name in names
            if name != "delegate_task"
        )
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

    def create(self, period_seconds: int, skill_name: str, instruction: str) -> dict:
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        LOGGER.info(
            "native-task-tool CREATE ARG period_seconds=%r type=%s skill=%r",
            repr(period_seconds)[:80], type(period_seconds).__name__, skill_name,
        )
        normalized_period = _period_seconds(period_seconds)
        if normalized_period is None:
            LOGGER.warning(
                "native-task-tool CREATE REJECT period_seconds=%r type=%s",
                repr(period_seconds)[:80], type(period_seconds).__name__,
            )
            return {"ok": False, "error": "period_seconds must be an integer in 10..604800"}
        period_seconds = normalized_period
        if not isinstance(skill_name, str) or skill_name not in self.allowed_skills:
            return {"ok": False, "error": "skill is not available to native AGENT"}
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > MAX_INSTRUCTION_CHARS:
            return {"ok": False, "error": "instruction must be 1..2048 characters"}
        if self._create_attempted:
            return {"ok": False, "error": "Already attempted creating a task in this turn; list tasks before retrying"}
        self._create_attempted = True
        instruction = instruction.strip()
        description = TASK_PREFIX + instruction[:100]
        # The schedule belongs to SYSTEM, not the model or an AGENT shell.
        # Every timer tick starts an independently bounded single AGENT run.
        task_text = (
            "Выполни ОДИН раз следующее действие через нативный инструмент "
            + skill_name + ". Не организуй циклы, ожидание, sleep, cron или "
            "собственное расписание. Следующий запуск сделает TASK SYSTEM.\n"
            "Действие: " + instruction
        )
        LOGGER.info(
            "native-task-tool CREATE period=%s skill=%s instruction=%r",
            period_seconds, skill_name, instruction,
        )
        try:
            record = self.rpc.call(
                "system", "create_periodic_task",
                description, task_text, [skill_name], float(period_seconds),
                method="query", executor="litert",
            )
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("CREATE", exc)
        LOGGER.info("native-task-tool CREATED id=%s period=%s skill=%s",
                    record["task_id"], period_seconds, skill_name)
        return {
            "ok": True, "task_id": record["task_id"],
            "period_seconds": period_seconds, "skill": skill_name,
            "status": "scheduled", "delivery": "TASK QUERY notifications inbox",
        }

    def list_queries(self) -> dict:
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
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
            }
            for r in records if self._ours(r)
        ]
        return {"ok": True, "tasks": entries}

    def stop(self, task_id: int) -> dict:
        if not self._allowed():
            return {"ok": False, "error": "task_management_not_in_active_turn"}
        if type(task_id) is not int or task_id < 1:
            return {"ok": False, "error": "task_id must be a positive integer"}
        try:
            record = self.rpc.call("tasks", "get", task_id)
            if record is None or not self._ours(record):
                return {"ok": False, "error": "native periodic QUERY not found"}
            stopped = self.rpc.call("system", "stop_task", task_id)
        except (TaskStoreError, OSError, ValueError) as exc:
            return self._rpc_error("STOP", exc)
        LOGGER.info("native-task-tool STOPPED id=%s", task_id)
        return {"ok": True, "task_id": task_id, "enabled": stopped["enabled"]}


def create_task_tools(controller: NativeTaskController) -> list:
    def create_periodic_query(period_seconds: int, skill_name: str, instruction: str) -> dict:
        """Создать ПЕРИОДИЧЕСКИЙ QUERY в долговременном TASK SYSTEM.

        Используй, когда пользователь просит выполнять действие каждые N
        секунд или минут и сообщать результат. ЭТО НЕ delegate_task:
        delegate_task выполняет только однократное поручение.
        Раз в period_seconds TASK SYSTEM запускает AGENT один раз, а
        результат публикует через QUERY MANAGER в уведомлениях native TUI.
        Нельзя использовать Shell-циклы, sleep или cron вместо этого.
        Выбери skill_name из нативных инструментов AGENT.

        Args:
            period_seconds (int): Интервал в секундах. Раз в минуту = 60.
            skill_name (str): Имя нужного инструмента AGENT, например get_outdoor_weather.
            instruction (str): Конкретное ОДНО измерение/действие; без циклов.
        """
        return controller.create(period_seconds, skill_name, instruction)

    def list_periodic_queries() -> dict:
        """Получить список периодических QUERY, созданных этим LiteRT MANAGER."""
        return controller.list_queries()

    def stop_periodic_query(task_id: int) -> dict:
        """Остановить ранее созданный периодический QUERY по его номеру.

        Args:
            task_id (int): Номер TASK из create_periodic_query или list_periodic_queries.
        """
        return controller.stop(task_id)

    return [create_periodic_query, list_periodic_queries, stop_periodic_query]
