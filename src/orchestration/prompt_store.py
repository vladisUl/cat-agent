from __future__ import annotations

import json
from pathlib import Path

from .skills import Skill


class PromptStore:
    MANAGER_TOOLS_MARKER = "**СПИСОК TOOLS:**"
    MQTT_MARKER = "**Список топиков и полей mqtt:**"

    def __init__(self, prompt_dir: Path, agent_count: int) -> None:
        self.prompt_dir = prompt_dir
        self.agent_count = agent_count
        self.prompt_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        required = [
            self.prompt_dir / "sys_prompt_manager.md",
            self.prompt_dir / "mqtt.md",
        ]
        required.extend(
            self.prompt_dir / f"sys_prompt_agent_{index}.md"
            for index in range(1, self.agent_count + 1)
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing prompt files: " + ", ".join(missing))

        manager = self._read("sys_prompt_manager.md")
        self._require_single_marker(manager, self.MANAGER_TOOLS_MARKER)
        self._require_single_marker(manager, self.MQTT_MARKER)

    def manager_system_prompt(self, tools: tuple[Skill, ...]) -> str:
        text = self._read("sys_prompt_manager.md")
        tool_text = self._tool_text(tools)
        text = self._insert_after_marker(
            text,
            self.MANAGER_TOOLS_MARKER,
            tool_text,
        )
        mqtt = self._read("mqtt.md")
        text = self._insert_after_marker(
            text,
            self.MQTT_MARKER,
            mqtt,
        )
        return text.strip()

    def agent_system_prompt(self, agent_id: str) -> str:
        index = self._agent_index(agent_id)
        return self._read(f"sys_prompt_agent_{index}.md")

    def write_manager_prompt(self, text: str) -> Path:
        path = self.prompt_dir / "prompt_manager.md"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return path

    def build_agent_bootstrap(
        self,
        tools: tuple[Skill, ...],
        workspace: Path,
    ) -> str:
        sections = [
            f"**WORKSPACE:** {workspace}",
            "**СПИСОК TOOLS:**",
            self._tool_text(tools),
        ]
        if any(tool.name == "mqtt" for tool in tools):
            sections.extend(
                [
                    self.MQTT_MARKER,
                    self._read("mqtt.md"),
                ]
            )
        return "\n\n".join(section for section in sections if section).strip() + "\n"

    def build_agent_system_context(
        self,
        agent_id: str,
        tools: tuple[Skill, ...],
        workspace: Path,
    ) -> str:
        system_prompt = self.agent_system_prompt(agent_id).strip()
        bootstrap = self.build_agent_bootstrap(tools, workspace).strip()
        return f"{system_prompt}\n\n{bootstrap}"

    @staticmethod
    def build_agent_task(task: str, method: str | None = None) -> str:
        payload: dict[str, str] = {}
        if method is not None:
            payload["method"] = method
        payload["task"] = task.strip()
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @staticmethod
    def build_agent_context(context: str) -> str:
        return json.dumps(
            {"context": context.strip()},
            ensure_ascii=False,
            indent=2,
        ) + "\n"

    def build_agent_prompt(
        self,
        agent_id: str,
        task: str,
        tools: tuple[Skill, ...],
        workspace: Path,
        *,
        method: str | None = None,
    ) -> str:
        system_context = self.build_agent_system_context(agent_id, tools, workspace)
        task_prompt = self.build_agent_task(task, method)
        text = system_context.rstrip() + "\n\n" + task_prompt
        self.write_agent_prompt(agent_id, text)
        return text

    def write_agent_prompt(self, agent_id: str, text: str) -> Path:
        index = self._agent_index(agent_id)
        path = self.prompt_dir / f"prompt_agent_{index}.md"
        desired = text.rstrip() + "\n"
        if path.is_file() and path.read_text(encoding="utf-8") == desired:
            return path
        path.write_text(desired, encoding="utf-8")
        return path

    @staticmethod
    def _tool_text(tools: tuple[Skill, ...]) -> str:
        return "\n\n".join(
            tool.prompt.strip()
            for tool in tools
            if tool.prompt.strip()
        )

    @staticmethod
    def _require_single_marker(text: str, marker: str) -> None:
        count = text.count(marker)
        if count != 1:
            raise ValueError(
                f"Prompt marker {marker!r} must appear exactly once, found {count}"
            )

    @classmethod
    def _insert_after_marker(cls, text: str, marker: str, payload: str) -> str:
        cls._require_single_marker(text, marker)
        before, after = text.split(marker, 1)
        payload = payload.strip()
        insertion = marker
        if payload:
            insertion += "\n\n" + payload
        return before + insertion + after

    def _read(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8").strip()

    @staticmethod
    def _agent_index(agent_id: str) -> int:
        if not agent_id.startswith("agent") or not agent_id[5:].isdigit():
            raise ValueError(f"Invalid agent id: {agent_id!r}")
        return int(agent_id[5:])
