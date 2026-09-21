"""Shared reserved-tool dispatch before either manager or agent reaches shell."""
from __future__ import annotations

import json


class ToolDispatcher:
    def __init__(self, mcp_runtime=None) -> None:
        self.mcp_runtime = mcp_runtime

    @staticmethod
    def is_mcp(command: str) -> bool:
        return command.lstrip().startswith("mcp:")

    def dispatch(self, command: str, runtime) -> str | None:
        # None means a legacy command: caller keeps its existing execution path.
        if not self.is_mcp(command):
            return None
        try:
            name, raw = command.strip().split(maxsplit=1)
            arguments = json.loads(raw, parse_constant=self._reject_constant,
                                   object_pairs_hook=self._unique_keys)
            if not isinstance(arguments, dict):
                raise ValueError("MCP arguments must be a JSON object")
            if not self._assigned(name, runtime.skill_names):
                raise ValueError("MCP tool is unknown or not assigned to this runtime")
            if self.mcp_runtime is None:
                raise ValueError("MCP is not configured")
        except (ValueError, RecursionError) as exc:
            return f"SYSTEM_ERROR\nMCP invalid_call: {exc}"
        key = name + " " + json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        uncertain = getattr(runtime, "_uncertain_commands", set())
        if key in uncertain:
            return "SYSTEM_ERROR\nMCP outcome_unknown: previous attempt may have executed; automatic replay refused"
        result = self.mcp_runtime.call(name, arguments)
        if result.startswith("SYSTEM_ERROR\nMCP outcome_unknown:"):
            uncertain.add(key)
            runtime._uncertain_commands = uncertain
        return result

    @staticmethod
    def _assigned(name: str, skill_names) -> bool:
        if name in skill_names:
            return True
        parts = name.split(":", 2)
        return (
            len(parts) == 3
            and parts[0] == "mcp"
            and f"mcp:{parts[1]}" in skill_names
        )

    @staticmethod
    def _reject_constant(value):
        raise ValueError("JSON must not contain NaN or Infinity")

    @staticmethod
    def _unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON contains duplicate keys")
            result[key] = value
        return result
