"""Immutable per-CORE union of prompt tools and startup-discovered MCP tools."""
from __future__ import annotations

from .skills import Skill, SkillBase, SkillBaseError


class ToolCatalog:
    def __init__(self, base: SkillBase, mcp_runtime=None) -> None:
        self.path = base.path
        self.mcp_runtime = mcp_runtime
        self._skills = {name: base.get(name) for name in base.names()}
        self._mcp_skills = () if mcp_runtime is None else mcp_runtime.skills()
        for skill in self._mcp_skills:
            if skill.name in self._skills:
                raise SkillBaseError(f"Duplicate tool: {skill.name}")
            self._skills[skill.name] = skill

    def names(self) -> tuple[str, ...]:
        return tuple(self._skills)

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillBaseError(f"Unknown skill: {name}") from exc

    def require(self, names: tuple[str, ...]) -> tuple[Skill, ...]:
        return tuple(self.get(name) for name in names)

    def catalog_text(self) -> str:
        return "\n".join(f"{s.name} — {s.description}" for s in self._skills.values())

    def mcp_prompt(self) -> str:
        if not self._mcp_skills:
            return ""
        return (
            "MCP tools доступны через /work#mcp:<server>:<tool> {JSON}. "
            "После команды жди фактический результат runtime. "
            "Для TASK указывай полное имя MCP tool в списке tools. "
            "При outcome_unknown не повторяй действие: оно могло выполниться.\n"
            + "\n".join(s.prompt for s in self._mcp_skills)
        )

    def close(self) -> None:
        if self.mcp_runtime is not None:
            self.mcp_runtime.close()


def build_tool_catalog(path, configs):
    base = SkillBase(path)
    enabled = tuple(c for c in configs if c.enabled)
    if not enabled:
        return base
    from .mcp_runtime import McpRuntime
    runtime = McpRuntime(enabled)
    runtime.start()
    try:
        return ToolCatalog(base, runtime)
    except BaseException:
        runtime.close()
        raise
