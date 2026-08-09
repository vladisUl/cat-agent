from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


class SkillBaseError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    prompt: str


_HEADER_RE = re.compile(r"^\[SKILL ([a-z][a-z0-9_-]*)\]$")


class SkillBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._skills: dict[str, Skill] = {}
        self.reload()

    def reload(self) -> None:
        text = self.path.read_text(encoding="utf-8")
        self._skills = self._parse(text)
        if not self._skills:
            raise SkillBaseError(f"No skills found in {self.path}")

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
        return "\n".join(f"{skill.name} — {skill.description}" for skill in self._skills.values())

    @staticmethod
    def _parse(text: str) -> dict[str, Skill]:
        lines = text.splitlines()
        skills: dict[str, Skill] = {}
        index = 0

        while index < len(lines):
            line = lines[index].strip()
            index += 1
            if not line or line.startswith("#"):
                continue

            match = _HEADER_RE.fullmatch(line)
            if not match:
                raise SkillBaseError(f"Expected [SKILL name], got: {line!r}")
            header_name = match.group(1)

            block: list[str] = []
            while index < len(lines) and lines[index].strip() != "[/SKILL]":
                block.append(lines[index])
                index += 1
            if index >= len(lines):
                raise SkillBaseError(f"Skill {header_name!r} has no [/SKILL]")
            index += 1

            skill = SkillBase._parse_block(header_name, block)
            if skill.name in skills:
                raise SkillBaseError(f"Duplicate skill: {skill.name}")
            skills[skill.name] = skill

        return skills

    @staticmethod
    def _parse_block(header_name: str, lines: list[str]) -> Skill:
        sections: dict[str, list[str]] = {"description": [], "prompt": []}
        name: str | None = None
        current: str | None = None

        for raw in lines:
            stripped = raw.strip()
            if stripped.startswith("name:"):
                if current is not None:
                    raise SkillBaseError(f"name must precede text in skill {header_name}")
                name = stripped[len("name:") :].strip()
                continue
            if stripped == "description:":
                current = "description"
                continue
            if stripped == "prompt:":
                current = "prompt"
                continue
            if current is not None:
                sections[current].append(raw)
            elif stripped:
                raise SkillBaseError(f"Unexpected line in skill {header_name}: {raw!r}")

        if name != header_name:
            raise SkillBaseError(
                f"Skill header/name mismatch: header={header_name!r}, name={name!r}"
            )

        description = "\n".join(sections["description"]).strip()
        prompt = "\n".join(sections["prompt"]).strip()
        if not description:
            raise SkillBaseError(f"Skill {name!r} has empty description")
        if not prompt:
            raise SkillBaseError(f"Skill {name!r} has empty prompt")
        return Skill(name=name, description=description, prompt=prompt)
