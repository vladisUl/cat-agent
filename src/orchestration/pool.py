from __future__ import annotations

from .agent import AgentState, AgentWorker


class AgentPool:
    def __init__(self, workers: list[AgentWorker]) -> None:
        self._workers = {worker.agent_id: worker for worker in workers}

    def acquire(self) -> AgentWorker | None:
        for worker in self._workers.values():
            if worker.state is AgentState.FREE:
                return worker
        return None

    def get(self, agent_id: str) -> AgentWorker | None:
        return self._workers.get(agent_id)

    def status_text(self) -> str:
        return "\n".join(f"{worker.agent_id} {worker.state.value}" for worker in self._workers.values())
