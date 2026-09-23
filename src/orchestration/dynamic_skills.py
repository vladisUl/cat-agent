from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .skills import Skill, SkillBaseError


_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_RESERVED_NAMES = {
    "mqtt_pub",
    "mqtt_sub",
    "query_timer",
    "read_pic",
    "task_timer",
    "timer",
}


@dataclass(frozen=True, slots=True)
class DynamicSkill:
    name: str
    code: str
    description: str
    manager: bool

    def as_skill(self) -> Skill:
        return Skill(
            name=self.name,
            description=self.description,
            prompt=(
                f"Для выполнения skill {self.name} используй команду:\n"
                f"{self.code}\n"
                "Если skill требует входных данных, добавляй их после команды в форме, "
                "которую задаёт description; runtime передаст этот хвост сценарию как "
                "входной текст без интерпретации. После команды жди фактический результат "
                "runtime и продолжай задачу по полученным данным."
            ),
        )


def load_dynamic_skills(directory: Path | None) -> tuple[DynamicSkill, ...]:
    if directory is None or not directory.exists():
        return ()
    if not directory.is_dir():
        raise SkillBaseError(f"Dynamic skills path is not a directory: {directory}")

    skills: list[DynamicSkill] = []
    names: set[str] = set()
    codes: set[str] = set()

    for path in sorted(directory.glob("*.txt"), key=lambda item: item.name.casefold()):
        name = path.stem
        if not _NAME_RE.fullmatch(name):
            raise SkillBaseError(f"Invalid dynamic skill filename: {path.name!r}")
        if name in _RESERVED_NAMES:
            raise SkillBaseError(f"Reserved dynamic skill name: {name}")
        if path.is_symlink() or not path.is_file():
            raise SkillBaseError(f"Dynamic skill must be a regular file: {path}")

        fields = _parse_file(path)
        expected = {"code", "description", "manager"}
        unknown = set(fields) - expected
        missing = expected - set(fields)
        if unknown:
            raise SkillBaseError(
                f"Unknown fields in dynamic skill {path.name}: {sorted(unknown)}"
            )
        if missing:
            raise SkillBaseError(
                f"Missing fields in dynamic skill {path.name}: {sorted(missing)}"
            )

        code = fields["code"]
        description = fields["description"]
        manager_raw = fields["manager"].casefold()

        expected_code = f"/work#{name}.sh"
        if code != expected_code:
            raise SkillBaseError(
                f"Dynamic skill {name!r} code must be exactly {expected_code!r}"
            )
        if not description:
            raise SkillBaseError(f"Dynamic skill {name!r} has empty description")
        if manager_raw not in {"true", "false"}:
            raise SkillBaseError(
                f"Dynamic skill {name!r} manager must be true or false"
            )

        if name in names:
            raise SkillBaseError(f"Duplicate dynamic skill: {name}")
        if code in codes:
            raise SkillBaseError(f"Duplicate dynamic skill code: {code}")
        names.add(name)
        codes.add(code)
        skills.append(
            DynamicSkill(
                name=name,
                code=code,
                description=description,
                manager=manager_raw == "true",
            )
        )

    return tuple(skills)


def _parse_file(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise SkillBaseError(
                f"Invalid dynamic skill line {path.name}:{line_number}: {raw!r}"
            )
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in fields:
            raise SkillBaseError(
                f"Duplicate or empty field in {path.name}:{line_number}: {key!r}"
            )
        fields[key] = value
    return fields
