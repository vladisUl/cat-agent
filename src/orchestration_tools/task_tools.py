"""Native MANAGER tools for durable periodic QUERY tasks in shared TASK SYSTEM.

A native tool may run synchronously inside send_message(); it must never
perform nested inference or create a local timer. Only authoritative RPC
mutates persistent task state. A lost RPC reply is an UNKNOWN outcome.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from litert_agent_tools.record_tools import RECORD_TOOL_NAME, extract_record_filename
from orchestration.tasks import TaskStoreError

LOGGER = logging.getLogger(__name__)
MAX_PERIOD_SECONDS = 604800
MAX_INSTRUCTION_CHARS = 2048
TASK_PREFIX = "LiteRT QUERY: "
_FILE_WRITE_INTENT = re.compile(
    r"запис|запиш|допис|допиш|сохран|файл|write|append|save|file",
    re.IGNORECASE,
)


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


_READONLY_MQTT_ACTIONS = {
    "get_outdoor_weather": "Измерь текущую температуру и влажность на улице",
    "get_home_climate": "Измерь текущую температуру и влажность дома",
    "get_aquarium_temperature": "Измерь текущую температуру воды в аквариуме",
    "get_freezer_temperature": "Измерь текущую температуру в морозильнике",
}


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
        self._human_request = ""

    def begin_turn(self, human_request: str = "") -> None:
        if self.in_turn:
            raise RuntimeError("Nested native TASK management turn")
        self.in_turn = True
        self._create_attempted = False
        self._human_request = human_request if isinstance(human_request, str) else ""

    def finish_turn(self) -> None:
        self.in_turn = False
        self._human_request = ""

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
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > MAX_INSTRUCTION_CHARS:
            return {"ok": False, "error": "instruction must be 1..2048 characters"}
        # Validate against the *original human request* as well: the model
        # previously erased the requested file write when choosing a skill.
        request_text = self._human_request + "\n" + instruction
        wants_file = bool(_FILE_WRITE_INTENT.search(request_text))
        filename = None
        if wants_file or skill_name == RECORD_TOOL_NAME:
            filename = extract_record_filename(request_text)
            if filename is None:
                return {"ok": False, "error": "File recording requires one safe .txt filename"}
            if skill_name not in ("get_outdoor_weather", RECORD_TOOL_NAME):
                return {"ok": False, "error": "Outdoor file recording requires record_outdoor_weather"}
            if RECORD_TOOL_NAME not in self.allowed_skills:
                return {"ok": False, "error": "record_outdoor_weather is not enabled for an AGENT"}
            LOGGER.info("native-task-tool FILE ROUTE skill=%s -> %s filename=%s",
                        skill_name, RECORD_TOOL_NAME, filename)
            skill_name = RECORD_TOOL_NAME
        elif not isinstance(skill_name, str) or skill_name not in self.allowed_skills:
            return {"ok": False, "error": "skill is not available to native AGENT"}
        if self._create_attempted:
            return {"ok": False, "error": "Already attempted creating a task in this turn; list tasks before retrying"}
        self._create_attempted = True
        instruction = instruction.strip()
        description = TASK_PREFIX + instruction[:100]
        # The schedule belongs to SYSTEM, not the model or an AGENT shell.
        # Every timer tick starts an independently bounded single AGENT run.
        if skill_name == RECORD_TOOL_NAME:
            task_text = (
                "СЕЙЧАС ОДИН раз вызови нативный инструмент "
                f"record_outdoor_weather(file_name={filename!r}). "
                "Этот инструмент сам считывает MQTT и дописывает одну "
                "строку в файл. НИКОГДА не сообщай об успешной записи "
                "без ok=true в фактическом результате инструмента. "
                "Не используй shell, циклы, sleep или cron. "
                "Следующий запуск назначит TASK SYSTEM. "
                "Сообщи путь файла и измеренные значения из результата.\n"
                "Исходная просьба: " + request_text
            )
        else:
            action = _READONLY_MQTT_ACTIONS.get(skill_name, instruction)
            task_text = (
                "СЕЙЧАС выполни ОДИН раз: " + action + ". "
                "Обязательно вызови нативный инструмент " + skill_name + "(), "
                "дождись его результата, затем сообщи фактические значения "
                "или причину ошибки. Не отвечай до вызова инструмента. "
                "Не организуй циклы, ожидание, sleep, cron или собственное "
                "расписание: следующий запуск сделает TASK SYSTEM.\n"
                "Дополнительная инструкция: " + instruction
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
        result = {
            "ok": True, "task_id": record["task_id"],
            "period_seconds": period_seconds, "skill": skill_name,
            "status": "scheduled", "delivery": "TASK QUERY notifications inbox",
        }
        if filename is not None:
            result["file_name"] = filename
        return result

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
    def create_periodic_query(period_seconds: int, skill_name: str, instruction: str) -> dict:
        """Создать ПЕРИОДИЧЕСКИЙ QUERY в долговременном TASK SYSTEM.

        Используй, когда пользователь просит выполнять действие каждые N
        секунд или минут и сообщать результат. ЭТО НЕ delegate_task:
        delegate_task выполняет только однократное поручение.
        Раз в period_seconds TASK SYSTEM запускает AGENT один раз, а
        результат публикует через QUERY MANAGER в уведомлениях native TUI.
        Нельзя использовать Shell-циклы, sleep или cron вместо этого.
        Выбери skill_name из нативных инструментов AGENT.
        Для температуры на улице + сохранения в .txt используй
        record_outdoor_weather и укажи имя файла в instruction.
        Одного get_outdoor_weather для записи файла недостаточно.

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
