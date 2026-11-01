"""Strict declarative agent identities, prompts and native tool allowlists."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

MAX_AGENTS = 8
AGENT_ID = re.compile(r"^agent_[1-9][0-9]*$")


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    prompt_file: Path
    tool_names: tuple[str, ...]
    inherit_shared_tools: bool = True


def load_agent_specs(
    path: Path, *, prompt_dir: Path | None = None,
) -> tuple[AgentSpec, ...]:
    path = Path(path)
    root = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(root, dict) or set(root) != {"agents"} or not isinstance(root["agents"], list):
        raise ValueError("Agent config requires only the 'agents' array")
    entries = root["agents"]
    if not 1 <= len(entries) <= MAX_AGENTS:
        raise ValueError(f"Agent count must be 1..{MAX_AGENTS}")
    specs: list[AgentSpec] = []
    used: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not (
            {'id', 'prompt', 'tools'} <= set(entry)
            <= {'id', 'prompt', 'tools', 'inherit_shared_tools'}
        ):
            raise ValueError('Agent entry requires id, prompt, tools and optional inherit_shared_tools')
        inherit = entry.get('inherit_shared_tools', True)
        if type(inherit) is not bool:
            raise ValueError('inherit_shared_tools must be boolean')
        ident, prompt, tools = entry["id"], entry["prompt"], entry["tools"]
        if not isinstance(ident, str) or not AGENT_ID.fullmatch(ident) or ident in used:
            raise ValueError(f"Invalid/duplicate agent id: {ident!r}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{ident}: prompt must be a path")
        if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
            raise ValueError(f"{ident}: tools must be an array of names")
        if len(set(tools)) != len(tools):
            raise ValueError(f"{ident}: duplicate tool names")
        prompt_path = Path(prompt)
        if not prompt_path.is_absolute():
            prompt_path = (Path(prompt_dir) if prompt_dir is not None else path.parent) / prompt_path
        if not prompt_path.is_file():
            raise ValueError(f"{ident}: prompt not found: {prompt_path}")
        specs.append(AgentSpec(ident, prompt_path.resolve(), tuple(tools), inherit))
        used.add(ident)
    return tuple(specs)


def effective_agent_tool_names(
    spec: AgentSpec, builtins: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Common read-only tools plus explicitly granted tools, without duplicates.

    Native Conversation and TASK worker must use this same calculation.
    """
    return tuple(dict.fromkeys((*(builtins if spec.inherit_shared_tools else ()), *spec.tool_names)))
