"""One-turn deferred agent delegation without nested LiteRT inference.

The native callback only queues validated work. Manager inference ends before
an agent Conversation is allowed to run on the shared Engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging

LOGGER = logging.getLogger(__name__)
MAX_DELEGATIONS_PER_TURN = 3
MAX_TASK_CHARS = 4096


@dataclass(frozen=True, slots=True)
class PendingDelegation:
    call_id: str
    agent_id: str
    task: str


class DelegationBroker:
    def __init__(self, agent_ids: tuple[str, ...]):
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError("Duplicate agent IDs")
        self.agent_ids = frozenset(agent_ids)
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

    def enqueue(self, agent_id: str, task: str) -> dict:
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
    def delegate_task(agent_id: str, task: str) -> dict:
        """Передать одно задание указанному AGENT для выполнения после текущего хода.

        Это только постановка поручения в очередь. Результат пока НЕ получен.
        После выполнения AGENT система отдельно доставит тебе фактический
        результат; окончательный ответ пользователю формируй по нему.

        Args:
            agent_id (str): Имя доступного агента, например agent_1.
            task (str): Точный текст одного задания для агента.
        """
        return broker.enqueue(agent_id, task)

    return delegate_task
