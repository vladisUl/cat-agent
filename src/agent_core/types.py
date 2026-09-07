from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Callable

@dataclass(frozen=True, slots=True)
class InferenceTiming:
    phase: str
    phase_started: float | None
    prefill_seconds: float | None
    generation_seconds: float | None
    total_seconds: float | None
    finished_at: float | None


class ModelClient(Protocol):
    label: str
    def chat(self, messages: list[dict[str, str]]): ...
    def reset_to_base(self, messages: list[dict[str, str]]) -> None: ...
    def set_event_handler(self, handler: Callable | None) -> None: ...
    def fork(self, label: str) -> ModelClient: ...
    def close(self) -> None: ...
