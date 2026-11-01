"""One-turn deferred agent delegation without nested LiteRT inference.

The native callback only queues validated work. Manager inference ends before
an agent Conversation is allowed to run on the shared Engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import re

from .task_tools import _period_seconds

LOGGER = logging.getLogger(__name__)
MAX_DELEGATIONS_PER_TURN = 3
MAX_TASK_CHARS = 4096


@dataclass(frozen=True, slots=True)
class PendingDelegation:
    call_id: str
    agent_id: str
    task: str


class DelegationBroker:
    def __init__(self, agent_ids: tuple[str, ...], periodic_controller=None):
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("Duplicate agent IDs")
        self.agent_ids = frozenset(agent_ids)
        self.periodic_controller = periodic_controller
        self.enabled = False  # Gate remains closed during every BASE warm-up.
        self._turn_active = False
        self._count = 0
        self._pending: list[PendingDelegation] = []

    def begin_turn(self) -> None:
        if self._turn_active:
            raise RuntimeError("Nested manager turns are forbidden")
        self._turn_active = True
        self._pending.clear()
        self._count = 0

    def finish_turn(self) -> None:
        self._turn_active = False
        self._pending.clear()

    def take_pending(self) -> tuple[PendingDelegation, ...]:
        pending = tuple(self._pending)
        self._pending.clear()
        return pending

    def enqueue(self, agent_id: str, task: str, period_seconds: int = 0,
                mqtt_tool: str = "") -> dict:
        if not self.enabled:
            return {"ok": False, "error": "delegation_disabled_during_warm"}
        if not self._turn_active:
            return {"ok": False, "error": "no_active_manager_turn"}
        if not isinstance(agent_id, str) or agent_id not in self.agent_ids:
            return {"ok": False, "error": "unknown_agent"}
        if not isinstance(task, str) or not task.strip() or len(task) > MAX_TASK_CHARS:
            return {"ok": False, "error": "invalid_task"}
        try:
            task.encode("utf-8", "strict")
        except UnicodeEncodeError:
            return {"ok": False, "error": "invalid_unicode"}
        # -1 is the legacy external-event sentinel, not a timer interval.
        # The named MQTT tool selects a validated topic/field from mqtt_tools.json.
        external = (
            (type(period_seconds) in (int, float) and period_seconds == -1)
            or (isinstance(period_seconds, str)
                and re.fullmatch(r"-1(?:\\.0+)?", period_seconds.strip()) is not None)
        )
        if external:
            if self.periodic_controller is None:
                return {"ok": False, "error": "external_TASK_SYSTEM_not_configured"}
            return self.periodic_controller.create_external(agent_id, task, mqtt_tool)
        if mqtt_tool:
            return {"ok": False, "error": "mqtt_tool_requires_period_minus_one"}
        # Zero keeps the original synchronous one-shot delegation unchanged.
        # LiteRT native argument decoding may supply "0" or "60" as a string.
        zero = (
            (type(period_seconds) in (int, float) and period_seconds == 0)
            or (isinstance(period_seconds, str)
                and re.fullmatch(r"0+(?:\.0+)?", period_seconds.strip()) is not None)
        )
        if not zero:
            interval = _period_seconds(period_seconds)
            if interval is None:
                return {"ok": False, "error": "period_seconds must be -1, 0 or an integer in 10..604800"}
            if self.periodic_controller is None:
                return {"ok": False, "error": "periodic_TASK_SYSTEM_not_configured"}
            return self.periodic_controller.create(interval, agent_id, task)
        if type(period_seconds) is bool:
            return {"ok": False, "error": "invalid_period_seconds"}
        if self._count >= MAX_DELEGATIONS_PER_TURN:
            return {"ok": False, "error": "delegation_limit_reached"}
        self._count += 1
        item = PendingDelegation(f"delegation_{self._count}", agent_id, task.strip())
        self._pending.append(item)
        LOGGER.info("native-delegation QUEUED id=%s agent=%s", item.call_id, agent_id)
        return {
            "ok": True, "status": "queued", "call_id": item.call_id,
            "agent_id": item.agent_id,
            "note": "The result is not ready. It will be delivered in a later turn.",
        }


def create_delegate_tool(broker: DelegationBroker):
    def delegate_task(agent_id: str, task: str, period_seconds: int = 0,
                      mqtt_tool: str = "") -> dict:
        """Передать AGENT полное задание без изменения текста.

        period_seconds=0: выполнить один раз через привычную очередь и
        сообщить результат MANAGER. Период >0: TASK SYSTEM запускает того
        же AGENT с тем же текстом каждые N секунд, результат — QUERY inbox.
        period_seconds=-1: сохранить внешний MQTT QUERY в TASK SYSTEM.
        Требуется mqtt_tool — имя датчика из config/mqtt_tools.json
        (get_corridor_motion или get_kitchen_leak). Срабатывает на изменениях
        указанного логического поля; AGENT получает исходное значение события.
        AGENT самостоятельно использует все разрешённые ему инструменты.

        Args:
            agent_id (str): Имя агента, например agent_1.
            task (str): Полный текст поручения без расписания и без ограничений на инструменты.
            period_seconds (int): -1 для события MQTT, 0 — однократно, 60 — раз в минуту.
            mqtt_tool (str): Имя MQTT-инструмента только при period_seconds=-1.
        """
        return broker.enqueue(agent_id, task, period_seconds, mqtt_tool)

    return delegate_task
