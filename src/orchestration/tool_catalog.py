"""Immutable per-CORE union of prompt tools and startup-discovered MCP tools."""
from __future__ import annotations

from pathlib import Path

from .skills import Skill, SkillBase, SkillBaseError


class ToolCatalog:
    def __init__(self, base: SkillBase, mcp_runtime=None) -> None:
        self.path = base.path
        self.mcp_runtime = mcp_runtime
        self._base_names = base.names()
        self._skills = {name: base.get(name) for name in self._base_names}
        self._mcp_skills = () if mcp_runtime is None else mcp_runtime.skills()
        self._configs = {
            config.name: config
            for config in (() if mcp_runtime is None else mcp_runtime.configs)
        }

        grouped: dict[str, list[Skill]] = {}
        for skill in self._mcp_skills:
            if skill.name in self._skills:
                raise SkillBaseError(f"Duplicate tool: {skill.name}")
            parts = skill.name.split(":", 2)
            if len(parts) != 3 or parts[0] != "mcp":
                raise SkillBaseError(f"Invalid MCP tool name: {skill.name}")
            self._skills[skill.name] = skill
            grouped.setdefault(parts[1], []).append(skill)

        self._server_skills = {
            server: tuple(items) for server, items in grouped.items()
        }
        self._capabilities: dict[str, Skill] = {}
        for server, config in self._configs.items():
            tools = self._server_skills.get(server, ())
            if not tools:
                continue
            name = f"mcp:{server}"
            if name in self._skills:
                raise SkillBaseError(f"Duplicate tool: {name}")
            description = config.description.strip() or f"MCP server {server}"
            capability = Skill(
                name=name,
                description=description,
                prompt=self._capability_prompt(name, tools),
            )
            self._capabilities[server] = capability
            self._skills[name] = capability

        manager_names = list(self._base_names)
        for server, config in self._configs.items():
            if config.manager:
                manager_names.extend(
                    skill.name for skill in self._server_skills.get(server, ())
                )
        self._manager_names = tuple(manager_names)

    @staticmethod
    def _capability_prompt(name: str, tools: tuple[Skill, ...]) -> str:
        return (
            f"Capability {name}. Доступные MCP tools:\n"
            + "\n".join(skill.prompt for skill in tools)
        )

    def names(self) -> tuple[str, ...]:
        """All assignable tools/capabilities, including legacy MCP leaf names."""
        return tuple(self._skills)

    def manager_names(self) -> tuple[str, ...]:
        """Tools the manager may execute directly."""
        return self._manager_names

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise SkillBaseError(f"Unknown skill: {name}") from exc

    def require(self, names: tuple[str, ...]) -> tuple[Skill, ...]:
        return tuple(self.get(name) for name in names)

    def catalog_text(self) -> str:
        lines = [
            f"{self._skills[name].name} — {self._skills[name].description}"
            for name in self._base_names
        ]
        for server, capability in self._capabilities.items():
            lines.append(f"{capability.name} — {capability.description}")
            if self._configs[server].manager:
                lines.extend(
                    f"{skill.name} — {skill.description}"
                    for skill in self._server_skills.get(server, ())
                )
        return "\n".join(lines)

    def mcp_prompt(self) -> str:
        if not self._capabilities:
            return ""
        lines = [
            "MCP capabilities доступны для ЗАДАНИЙ как mcp:<server>.",
            "При передаче mcp:<server> AGENT получает полный frozen набор tools этого сервера.",
            "Полные схемы ниже доступны Гене напрямую только для серверов manager=true.",
            "После прямой MCP-команды жди фактический результат runtime; при outcome_unknown не повторяй действие.",
        ]
        for server, capability in self._capabilities.items():
            config = self._configs[server]
            mode = "direct+agent" if config.manager else "agent-only"
            lines.append(f"{capability.name} — {capability.description} [{mode}]")
            if config.manager:
                lines.extend(
                    skill.prompt for skill in self._server_skills.get(server, ())
                )
        return "\n".join(lines)

    def write_snapshots(self, directory: Path) -> None:
        expected: dict[str, str] = {}
        for server, config in self._configs.items():
            tools = self._server_skills.get(server, ())
            description = config.description.strip() or f"MCP server {server}"
            lines = [
                "# Generated from MCP tools/list at CORE startup. Do not edit.",
                f"server: {server}",
                f"manager: {'true' if config.manager else 'false'}",
                f"description: {description}",
                f"tools: {len(tools)}",
                "",
            ]
            if tools:
                lines.append(self._capability_prompt(f"mcp:{server}", tools))
            else:
                lines.append("No tools were discovered for this server in the frozen CORE catalog.")
            expected[f"{server}.txt"] = "\n".join(lines).rstrip() + "\n"
        _sync_snapshot_dir(directory, expected)

    def close(self) -> None:
        if self.mcp_runtime is not None:
            self.mcp_runtime.close()


def _sync_snapshot_dir(directory: Path, expected: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.txt"):
        if path.name not in expected:
            path.unlink()
    for name, text in expected.items():
        path = directory / name
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        path.write_text(text, encoding="utf-8")


def build_tool_catalog(path, configs, *, snapshot_dir: Path | None = None):
    base = SkillBase(path)
    enabled = tuple(c for c in configs if c.enabled)
    if not enabled:
        if snapshot_dir is not None:
            _sync_snapshot_dir(snapshot_dir, {})
        return base
    from .mcp_runtime import McpRuntime
    runtime = McpRuntime(enabled)
    runtime.start()
    try:
        catalog = ToolCatalog(base, runtime)
        if snapshot_dir is not None:
            catalog.write_snapshots(snapshot_dir)
        return catalog
    except BaseException:
        runtime.close()
        raise
