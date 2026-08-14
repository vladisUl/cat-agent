from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import shlex


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
_TIMER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_MANAGER_CONTROL_WORDS = {"ASK", "WAIT", "REPLY"}


def _looks_like_manager_control_line(line: str) -> bool:
    text = line.strip()
    if text in _MANAGER_CONTROL_WORDS:
        return True
    return any(
        text.startswith(prefix)
        for prefix in ("DELEGATE ", "CONTINUE ", "SYSTEM ", "timer.sh ")
    )


def _positive_number(raw: str) -> bool:
    try:
        return float(raw) > 0
    except ValueError:
        return False


def _timer_name(raw: str) -> str | None:
    return raw if _TIMER_NAME_RE.fullmatch(raw) else None


def _timer_script_error() -> str:
    return (
        "invalid timer.sh syntax; use one of:\n"
        "timer.sh <period_seconds> [name]\n"
        "<future event task on following lines>\n"
        "timer.sh start [name]\n"
        "timer.sh stop [name]\n"
        "timer.sh period <period_seconds> [name]\n"
        "timer.sh delete [name]\n"
        "timer.sh list\n"
        "period_seconds must be positive; omitted name means default"
    )


def _parse_timer_script(first: str, body: str) -> ManagerDirective:
    try:
        parts = shlex.split(first)
    except ValueError as exc:
        return ManagerDirective(None, "", error=f"invalid timer.sh syntax: {exc}")

    if not parts or parts[0] != "timer.sh" or len(parts) < 2:
        return ManagerDirective(None, "", error=_timer_script_error())

    arg = parts[1]

    # Natural creation form: timer.sh 60 [name]
    if _positive_number(arg):
        if len(parts) not in {2, 3}:
            return ManagerDirective(None, "", error=_timer_script_error())
        name = "default" if len(parts) == 2 else parts[2]
        if _timer_name(name) is None or not body:
            return ManagerDirective(None, "", error=_timer_script_error())
        if any(_looks_like_manager_control_line(line) for line in body.splitlines()):
            return ManagerDirective(
                None,
                "",
                error=(
                    "timer.sh task must contain only the future event task; "
                    "do not embed another control command"
                ),
            )
        return ManagerDirective(
            ManagerAction.SYSTEM,
            body,
            system_command=f"TIMER SET {name} {arg}",
        )

    operation = arg.lower()

    if operation in {"start", "stop", "delete"}:
        if len(parts) not in {2, 3} or body:
            return ManagerDirective(None, "", error=_timer_script_error())
        name = "default" if len(parts) == 2 else parts[2]
        if _timer_name(name) is None:
            return ManagerDirective(None, "", error=_timer_script_error())
        return ManagerDirective(
            ManagerAction.SYSTEM,
            "",
            system_command=f"TIMER {operation.upper()} {name}",
        )

    if operation == "period":
        if body or len(parts) not in {3, 4}:
            return ManagerDirective(None, "", error=_timer_script_error())

        # Preferred natural form: timer.sh period 120 [name].
        # Also accept timer.sh period name 120 so a model can recover naturally.
        if len(parts) == 3 and _positive_number(parts[2]):
            period = parts[2]
            name = "default"
        elif len(parts) == 4 and _positive_number(parts[2]):
            period = parts[2]
            name = parts[3]
        elif len(parts) == 4 and _positive_number(parts[3]):
            name = parts[2]
            period = parts[3]
        else:
            return ManagerDirective(None, "", error=_timer_script_error())

        if _timer_name(name) is None:
            return ManagerDirective(None, "", error=_timer_script_error())
        return ManagerDirective(
            ManagerAction.SYSTEM,
            "",
            system_command=f"TIMER PERIOD {name} {period}",
        )

    if operation == "list":
        if len(parts) != 2 or body:
            return ManagerDirective(None, "", error=_timer_script_error())
        return ManagerDirective(ManagerAction.SYSTEM, "", system_command="TIMER LIST")

    return ManagerDirective(None, "", error=_timer_script_error())


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

    if first == "timer.sh" or first.startswith("timer.sh "):
        return _parse_timer_script(first, body)

    # Legacy internal form is still accepted for compatibility, but it is no
    # longer exposed to the model. timer.sh is the model-facing interface.
    if first.startswith("SYSTEM "):
        command = first[len("SYSTEM ") :].strip()
        if not command:
            return ManagerDirective(None, "", error="SYSTEM requires a command")
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
        error="first line must be DELEGATE, CONTINUE, timer.sh, ASK, WAIT, or REPLY",
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
