"""Proof of native side effects during one serialized AGENT task turn.

The worker records actual tool results, not the model's final text.
The context never survives into a later cloned Conversation/turn.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TaskToolAudit:
    results: dict[str, list[dict]] = field(default_factory=dict)

    def successful(self, tool_name: str) -> dict | None:
        return next(
            (record for record in reversed(self.results.get(tool_name, ()))
             if record.get("ok") is True),
            None,
        )

    def last(self, tool_name: str) -> dict | None:
        entries = self.results.get(tool_name, ())
        return entries[-1] if entries else None


_active: ContextVar[TaskToolAudit | None] = ContextVar(
    "native_task_tool_audit", default=None,
)


@contextmanager
def capture_task_tools() -> Iterator[TaskToolAudit]:
    audit = TaskToolAudit()
    token = _active.set(audit)
    try:
        yield audit
    finally:
        _active.reset(token)


def record_task_tool_result(name: str, result: dict) -> None:
    audit = _active.get()
    if audit is not None:
        audit.results.setdefault(name, []).append(dict(result))
