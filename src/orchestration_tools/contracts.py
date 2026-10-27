"""Initial cross-package contract; concrete model runtime comes in stage 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class ManagerTurn:
    kind: Literal["reply", "silent", "error"]
    text: str = ""


class ManagerRuntime(Protocol):
    """Future manager owns per-session Conversation; IPC never owns the model."""

    def respond(self, session_id: str, text: str) -> ManagerTurn: ...

    def release_session(self, session_id: str) -> None: ...

    def close(self) -> None: ...
