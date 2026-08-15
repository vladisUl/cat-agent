from __future__ import annotations

from pathlib import Path

from .skills import Skill

MANAGER_BOOTSTRAP_ACK = "READY"
AGENT_BOOTSTRAP_ACK = "READY"


class PromptStore:
    def __init__(self, prompt_dir: Path, agent_count: int) -> None:
        self.prompt_dir = prompt_dir
        self.agent_count = agent_count
        self.prompt_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        required = [self.prompt_dir / "sys_prompt_manager.txt", self.prompt_dir / "prompt_base.txt"]
        required.extend(
            self.prompt_dir / f"sys_prompt_agent_{index}.txt"
            for index in range(1, self.agent_count + 1)
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing prompt files: " + ", ".join(missing))

    def manager_system_prompt(self) -> str:
        return self._read("sys_prompt_manager.txt")

    def agent_system_prompt(self, agent_id: str) -> str:
        index = self._agent_index(agent_id)
        return self._read(f"sys_prompt_agent_{index}.txt")

    def write_manager_prompt(self, text: str) -> Path:
        path = self.prompt_dir / "prompt_manager.txt"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return path

    def build_agent_bootstrap(
        self,
        skills: tuple[Skill, ...],
        workspace: Path,
    ) -> str:
        parts = [
            "[WORKSPACE]",
            str(workspace),
            "[/WORKSPACE]",
        ]
        for skill in skills:
            parts.extend(
                [
                    "",
                    f"[SKILL {skill.name}]",
                    skill.prompt.strip(),
                    "[/SKILL]",
                ]
            )
            context = self._skill_context(skill.name)
            if context:
                parts.extend(
                    [
                        "",
                        f"[CONTEXT {skill.name}]",
                        context,
                        f"[/CONTEXT]",
                    ]
                )
        parts.extend(
            [
                "",
                "[BOOTSTRAP]",
                "Контекст агента загружен. Следующее сообщение user содержит TASK.",
                f"Подтверди инициализацию словом {AGENT_BOOTSTRAP_ACK}.",
                "[/BOOTSTRAP]",
            ]
        )
        return "\n".join(parts).rstrip() + "\n"

    @staticmethod
    def build_agent_task(task: str, method: str | None = None) -> str:
        parts: list[str] = []
        if method is not None:
            parts.extend(["[METHOD]", method.upper(), "[/METHOD]"])
        parts.extend(["[TASK]", task.strip(), "[/TASK]"])
        return "\n".join(parts) + "\n"

    def build_agent_prompt(
        self,
        agent_id: str,
        task: str,
        skills: tuple[Skill, ...],
        workspace: Path,
        *,
        method: str | None = None,
    ) -> str:
        bootstrap = self.build_agent_bootstrap(skills, workspace)
        task_prompt = self.build_agent_task(task, method)
        text = bootstrap.rstrip() + "\n\n" + task_prompt
        self.write_agent_prompt(agent_id, text)
        return text

    def write_agent_prompt(self, agent_id: str, text: str) -> Path:
        index = self._agent_index(agent_id)
        path = self.prompt_dir / f"prompt_agent_{index}.txt"
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return path

    def _skill_context(self, skill_name: str) -> str:
        path = self.prompt_dir / f"{skill_name}.txt"
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()

    def _read(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8").strip()

    @staticmethod
    def _agent_index(agent_id: str) -> int:
        if not agent_id.startswith("agent") or not agent_id[5:].isdigit():
            raise ValueError(f"Invalid agent id: {agent_id!r}")
        return int(agent_id[5:])
