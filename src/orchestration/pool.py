from __future__ import annotations

from .agent import AgentState, AgentWorker


class AgentPool:
    def __init__(
        self,
        workers: list[AgentWorker],
        *,
        event_worker_id: str | None = None,
    ) -> None:
        self._workers = {worker.agent_id: worker for worker in workers}
        if event_worker_id is not None and event_worker_id not in self._workers:
            raise ValueError(f"unknown event worker: {event_worker_id}")
        self._event_worker_id = event_worker_id

    def acquire(self) -> AgentWorker | None:
        for worker in self._workers.values():
            if worker.agent_id == self._event_worker_id:
                continue
            if worker.state is AgentState.FREE:
                return worker
        return None

    def acquire_event(self) -> AgentWorker | None:
        if self._event_worker_id is not None:
            worker = self._workers[self._event_worker_id]
            return worker if worker.state is AgentState.FREE else None
        return self.acquire()

    def get(self, agent_id: str) -> AgentWorker | None:
        return self._workers.get(agent_id)

    def status_text(self) -> str:
        return "\n".join(
            f"{worker.agent_id} {worker.state.value}"
            for worker in self._workers.values()
        )
