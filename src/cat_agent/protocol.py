from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class ManagerAction(str, Enum):
    DELEGATE = "DELEGATE"
    CONTINUE = "CONTINUE"
    SYSTEM = "SYSTEM"
    ASK = "ASK"
    WAIT = "WAIT"
    REPLY = "REPLY"


@dataclass(frozen=True, slots=True)
class ManagerDirective:
    action: ManagerAction | None
    body: str
    skills: tuple[str, ...] = ()
    agent_id: str | None = None
    system_command: str | None = None
    error: str | None = None


class AgentAction(str, Enum):
    COMMAND = "COMMAND"
    DONE = "DONE"
    NEED = "NEED"


@dataclass(frozen=True, slots=True)
class AgentDirective:
    action: AgentAction | None
    body: str
    command: str | None = None
    error: str | None = None


_SKILL_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_AGENT_RE = re.compile(r"^agent[1-9][0-9]*$")
_MANAGER_CONTROL_WORDS = {"ASK", "WAIT", "REPLY"}


def _looks_like_manager_control_line(line: str) -> bool:
    text = line.strip()
    if text in _MANAGER_CONTROL_WORDS:
        return True
    return any(
        text.startswith(prefix)
        for prefix in ("DELEGATE ", "CONTINUE ", "SYSTEM ")
    )


def parse_manager_output(content: str, *, max_chars: int = 8192) -> ManagerDirective:
    text = content.strip()
    if not text:
        return ManagerDirective(None, "", error="empty response")
    if len(text) > max_chars:
        return ManagerDirective(None, "", error=f"response is longer than {max_chars} characters")
    if text.startswith("```"):
        return ManagerDirective(None, "", error="Markdown code fences are not accepted")

    first, sep, rest = text.partition("\n")
    first = first.strip()
    body = rest.strip() if sep else ""

    if first == "ASK":
        if not body:
            return ManagerDirective(None, "", error="ASK requires a question on following lines")
        return ManagerDirective(ManagerAction.ASK, body)

    if first == "REPLY":
        if not body:
            return ManagerDirective(None, "", error="REPLY requires text on following lines")
        return ManagerDirective(ManagerAction.REPLY, body)

    if first == "WAIT":
        if body:
            return ManagerDirective(None, "", error="WAIT must not contain additional text")
        return ManagerDirective(ManagerAction.WAIT, "")

    if first.startswith("SYSTEM "):
        command = first[len("SYSTEM ") :].strip()
        if not command:
            return ManagerDirective(None, "", error="SYSTEM requires a command")

        parts = command.split()
        if len(parts) >= 2 and parts[0].upper() == "TIMER":
            operation = parts[1].upper()
            if operation == "SET":
                if not body:
                    return ManagerDirective(
                        None,
                        "",
                        error="SYSTEM TIMER SET requires only the future event task on following lines",
                    )
                if any(_looks_like_manager_control_line(line) for line in body.splitlines()):
                    return ManagerDirective(
                        None,
                        "",
                        error=(
                            "SYSTEM TIMER SET body must contain only the future event task; "
                            "do not embed DELEGATE, CONTINUE, SYSTEM, ASK, WAIT, or REPLY"
                        ),
                    )
            elif body:
                return ManagerDirective(
                    None,
                    "",
                    error=f"SYSTEM TIMER {operation} must not contain additional text",
                )

        return ManagerDirective(
            ManagerAction.SYSTEM,
            body,
            system_command=command,
        )

    if first.startswith("DELEGATE "):
        skills_text = first[len("DELEGATE ") :].strip()
        if not skills_text:
            return ManagerDirective(None, "", error="DELEGATE requires at least one skill")
        skills = tuple(item.strip() for item in skills_text.split(",") if item.strip())
        if not skills or any(not _SKILL_RE.fullmatch(item) for item in skills):
            return ManagerDirective(None, "", error="invalid DELEGATE skill list")
        if len(set(skills)) != len(skills):
            return ManagerDirective(None, "", error="DELEGATE contains duplicate skills")
        if not body:
            return ManagerDirective(None, "", error="DELEGATE requires a task on following lines")
        return ManagerDirective(ManagerAction.DELEGATE, body, skills=skills)

    if first.startswith("CONTINUE "):
        agent_id = first[len("CONTINUE ") :].strip()
        if not _AGENT_RE.fullmatch(agent_id):
            return ManagerDirective(None, "", error="CONTINUE requires an agent id such as agent1")
        if not body:
            return ManagerDirective(None, "", error="CONTINUE requires additional context")
        return ManagerDirective(ManagerAction.CONTINUE, body, agent_id=agent_id)

    return ManagerDirective(
        None,
        "",
        error="first line must be DELEGATE, CONTINUE, SYSTEM, ASK, WAIT, or REPLY",
    )


def parse_agent_output(content: str, *, max_chars: int = 8192) -> AgentDirective:
    text = content.strip()
    if not text:
        return AgentDirective(None, "", error="empty response")
    if len(text) > max_chars:
        return AgentDirective(None, "", error=f"response is longer than {max_chars} characters")
    if text.startswith("```") or text.endswith("```"):
        return AgentDirective(None, "", error="Markdown code fences are not accepted")

    first, sep, rest = text.partition("\n")
    first = first.strip()
    body = rest.strip() if sep else ""

    if first == "DONE":
        if not body:
            return AgentDirective(None, "", error="DONE requires a concise result")
        return AgentDirective(AgentAction.DONE, body)

    if first == "NEED":
        if not body:
            return AgentDirective(None, "", error="NEED requires a description of missing information")
        return AgentDirective(AgentAction.NEED, body)

    if "\n" in text or "\r" in text:
        return AgentDirective(
            None,
            "",
            error=(
                "previous response was rejected completely and no command was executed; "
                "exactly one action is allowed per response: one command on one line, "
                "or DONE with its result, or NEED with its reason"
            ),
        )

    return AgentDirective(AgentAction.COMMAND, "", command=text)
