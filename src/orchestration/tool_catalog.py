"""Frozen per-CORE catalog of local TOOL files and discovered MCP tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .skills import Skill, SkillBaseError


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class LocalTool:
    name: str
    manager: bool
    text: str
    description: str

    def as_skill(self) -> Skill:
        return Skill(
            name=self.name,
            description=self.description,
            prompt=self.text,
        )


def _plain_markdown(line: str) -> str:
    return line.strip().replace("**", "").strip()


def _field(text: str, name: str) -> str | None:
    prefix = name.casefold() + ":"
    for raw in text.splitlines():
        plain = _plain_markdown(raw)
        if plain.casefold().startswith(prefix):
            return plain[len(prefix):].strip()
    return None


def _description(text: str, fallback: str) -> str:
    lines = text.splitlines()
    for index, raw in enumerate(lines):
        plain = _plain_markdown(raw)
        if not plain.casefold().startswith("description:"):
            continue
        value = plain.split(":", 1)[1].strip()
        if value:
            return value
        for candidate in lines[index + 1:]:
            candidate_plain = _plain_markdown(candidate)
            if not candidate_plain:
                continue
            if candidate_plain.startswith("[") or candidate_plain.startswith(chr(96) * 3):
                continue
            if re.match(r"^[A-Za-zА-Яа-я_ -]+:\s*", candidate_plain):
                break
            return candidate_plain.lstrip("* ").strip() or fallback
        break
    return fallback


def _mcp_capability_prompt(
    name: str,
    description: str,
    tools: tuple[Skill, ...],
) -> str:
    lines = [
        f"**[TOOL {name}]**",
        "",
        f"**name:** {name}",
        "",
        f"**description:** {description}",
        "",
        "**tools:**",
    ]
    for tool in tools:
        detail = tool.description.strip()
        lines.append(
            f"- {tool.name} — {detail}" if detail else f"- {tool.name}"
        )
    lines.extend(["", "**[/TOOL]**"])
    return "\n".join(lines)


def load_tool_files(directory: Path) -> tuple[LocalTool, ...]:
    if not directory.exists():
        raise SkillBaseError(f"Tools directory does not exist: {directory}")
    if not directory.is_dir():
        raise SkillBaseError(f"Tools path is not a directory: {directory}")

    result: list[LocalTool] = []
    names: set[str] = set()

    for path in sorted(directory.glob("*.md"), key=lambda item: item.name.casefold()):
        if path.is_symlink() or not path.is_file():
            raise SkillBaseError(f"Tool definition must be a regular file: {path}")

        filename_name = path.stem
        if not _NAME_RE.fullmatch(filename_name):
            raise SkillBaseError(f"Invalid tool filename: {path.name!r}")

        text = path.read_text(encoding="utf-8").strip()
        nonempty = [line for line in text.splitlines() if line.strip()]
        if len(nonempty) < 2:
            raise SkillBaseError(f"Tool definition is incomplete: {path.name}")

        opening = _plain_markdown(nonempty[0])
        closing = _plain_markdown(nonempty[-1])
        expected_opening = f"[TOOL {filename_name}]"
        if opening != expected_opening:
            raise SkillBaseError(
                f"Tool header mismatch in {path.name}: expected {expected_opening!r}, got {opening!r}"
            )
        if closing != "[/TOOL]":
            raise SkillBaseError(f"Tool {filename_name!r} has no [/TOOL] closing tag")

        declared_name = _field(text, "name")
        if declared_name != filename_name:
            raise SkillBaseError(
                f"Tool filename/name mismatch: filename={filename_name!r}, name={declared_name!r}"
            )

        manager_raw = _field(text, "manager")
        if manager_raw is None or manager_raw.casefold() not in {"true", "false"}:
            raise SkillBaseError(
                f"Tool {filename_name!r} manager must be true or false"
            )

        if filename_name in names:
            raise SkillBaseError(f"Duplicate tool: {filename_name}")
        names.add(filename_name)
        result.append(
            LocalTool(
                name=filename_name,
                manager=manager_raw.casefold() == "true",
                text=text,
                description=_description(text, filename_name),
            )
        )

    if not result:
        raise SkillBaseError(f"No TOOL definitions found in {directory}")
    return tuple(result)


class ToolCatalog:
    def __init__(
        self,
        local_tools: tuple[LocalTool, ...],
        mcp_runtime=None,
    ) -> None:
        self.mcp_runtime = mcp_runtime
        self._local_tools = tuple(local_tools)
        self._tools = {spec.name: spec.as_skill() for spec in self._local_tools}

        self._mcp_skills = () if mcp_runtime is None else mcp_runtime.skills()
        self._configs = {
            config.name: config
            for config in (() if mcp_runtime is None else mcp_runtime.configs)
        }

        grouped: dict[str, list[Skill]] = {}
        for tool in self._mcp_skills:
            if tool.name in self._tools:
                raise SkillBaseError(f"Duplicate tool: {tool.name}")
            parts = tool.name.split(":", 2)
            if len(parts) != 3 or parts[0] != "mcp":
                raise SkillBaseError(f"Invalid MCP tool name: {tool.name}")
            self._tools[tool.name] = tool
            grouped.setdefault(parts[1], []).append(tool)

        self._server_tools = {
            server: tuple(items) for server, items in grouped.items()
        }
        self._capabilities: dict[str, Skill] = {}
        self._manager_capabilities: dict[str, Skill] = {}
        for server, config in self._configs.items():
            tools = self._server_tools.get(server, ())
            if not tools:
                continue
            name = f"mcp:{server}"
            if name in self._tools:
                raise SkillBaseError(f"Duplicate tool: {name}")
            description = config.description.strip() or f"MCP server {server}"

            # AGENT capability: assigning mcp:<server> expands to the complete
            # frozen leaf-tool set discovered at CORE startup.
            capability = Skill(
                name=name,
                description=description,
                prompt="\n\n".join(tool.prompt for tool in tools),
            )
            self._capabilities[server] = capability
            self._tools[name] = capability

            # MANAGER visibility is separate from direct-call authorization.
            # Every MCP server is visible as a compact delegation capability,
            # while only manager=true leaf tools are directly callable.
            self._manager_capabilities[server] = Skill(
                name=name,
                description=description,
                prompt=_mcp_capability_prompt(name, description, tools),
            )

        manager_names = [spec.name for spec in self._local_tools if spec.manager]
        manager_prompt_tools = [
            spec.as_skill() for spec in self._local_tools if spec.manager
        ]
        for server, config in self._configs.items():
            capability = self._manager_capabilities.get(server)
            if capability is not None:
                manager_prompt_tools.append(capability)
            if config.manager:
                leaf_tools = self._server_tools.get(server, ())
                manager_names.extend(tool.name for tool in leaf_tools)
                manager_prompt_tools.extend(leaf_tools)
        self._manager_names = tuple(manager_names)
        self._manager_prompt_tools = tuple(manager_prompt_tools)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def manager_names(self) -> tuple[str, ...]:
        """Tools MANAGER is authorized to invoke directly."""
        return self._manager_names

    def manager_prompt_tools(self) -> tuple[Skill, ...]:
        """Tools/capabilities MANAGER must know about in its frozen BASE."""
        return self._manager_prompt_tools

    def get(self, name: str) -> Skill:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise SkillBaseError(f"Unknown tool: {name}") from exc

    def require(self, names: tuple[str, ...]) -> tuple[Skill, ...]:
        return tuple(self.get(name) for name in names)

    def catalog_text(self) -> str:
        return "\n".join(
            f"{name} — {self._tools[name].description}"
            for name in self._tools
        )

    def write_snapshots(self, directory: Path) -> None:
        expected: dict[str, str] = {}
        for server, config in self._configs.items():
            tools = self._server_tools.get(server, ())
            description = config.description.strip() or f"MCP server {server}"
            lines = [
                "# Generated from MCP tools/list at CORE startup. Do not edit.",
                f"server: {server}",
                f"manager: {'true' if config.manager else 'false'}",
                f"description: {description}",
                f"tools: {len(tools)}",
                "",
            ]
            lines.extend(tool.prompt for tool in tools)
            expected[f"{server}.md"] = "\n".join(lines).rstrip() + "\n"
        _sync_snapshot_dir(directory, expected)

    def close(self) -> None:
        if self.mcp_runtime is not None:
            self.mcp_runtime.close()


def _sync_snapshot_dir(directory: Path, expected: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.md", "*.txt"):
        for path in directory.glob(pattern):
            if path.name not in expected:
                path.unlink()
    for name, text in expected.items():
        path = directory / name
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            continue
        path.write_text(text, encoding="utf-8")


def build_tool_catalog(
    tools_dir: Path,
    configs,
    *,
    snapshot_dir: Path | None = None,
):
    local_tools = load_tool_files(tools_dir)
    enabled = tuple(config for config in configs if config.enabled)

    if not enabled:
        if snapshot_dir is not None:
            _sync_snapshot_dir(snapshot_dir, {})
        return ToolCatalog(local_tools)

    from .mcp_runtime import McpRuntime

    runtime = McpRuntime(enabled)
    runtime.start()
    try:
        catalog = ToolCatalog(local_tools, runtime)
        if snapshot_dir is not None:
            catalog.write_snapshots(snapshot_dir)
        return catalog
    except BaseException:
        runtime.close()
        raise
