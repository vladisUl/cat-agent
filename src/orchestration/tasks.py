from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile


DEFAULT_TASK_FILE = Path("/var/lib/cat-agent/task.txt")
TASK_METHODS = frozenset({"task", "query"})


class TaskStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: int
    description: str
    text: str
    method: str = "task"
    skills: tuple[str, ...] = ()
    timer_period_seconds: float | None = None
    enabled: bool = True


class TaskStore:
    """Small persistent registry for long-lived tasks.

    Model/session state is deliberately absent here. A process can die, all KV
    state can disappear, and SYSTEM can reconstruct long-lived tasks from disk.
    """

    def __init__(self, path: Path = DEFAULT_TASK_FILE, *, max_tasks: int = 5) -> None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be >= 1")
        self.path = path
        self.max_tasks = max_tasks
        self._tasks: dict[int, TaskRecord] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.exists():
            self._tasks = {}
            return

        tasks: dict[int, TaskRecord] = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise TaskStoreError(f"cannot read task store {self.path}: {exc}") from exc

        for line_number, raw in enumerate(lines, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                task_id = int(item["task"])
                description = str(item["description"]).strip()
                text = str(item["text"]).strip()
                method = str(item.get("method", "task")).strip().lower()
                raw_skills = item.get("skills", [])
                if not isinstance(raw_skills, list):
                    raise TypeError("skills must be a list")
                skills = tuple(str(name).strip() for name in raw_skills)
                timer_raw = item.get("timer_period_seconds")
                timer_period_seconds = (
                    None if timer_raw is None else float(timer_raw)
                )
                enabled = bool(item.get("enabled", True))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise TaskStoreError(
                    f"invalid task record at {self.path}:{line_number}"
                ) from exc

            self._validate_record(
                task_id,
                description,
                text,
                method,
                skills,
                timer_period_seconds,
                line_number=line_number,
            )
            if task_id in tasks:
                raise TaskStoreError(
                    f"duplicate task id {task_id} at {self.path}:{line_number}"
                )

            tasks[task_id] = TaskRecord(
                task_id=task_id,
                description=description,
                text=text,
                method=method,
                skills=skills,
                timer_period_seconds=timer_period_seconds,
                enabled=enabled,
            )

        self._tasks = tasks

    def create(
        self,
        description: str,
        text: str,
        *,
        method: str = "task",
        skills: tuple[str, ...] = (),
        timer_period_seconds: float | None = None,
        enabled: bool = True,
    ) -> TaskRecord:
        description = description.strip()
        text = text.strip()
        method = method.strip().lower()
        skills = tuple(name.strip() for name in skills)
        self._validate_record(
            1,
            description,
            text,
            method,
            skills,
            timer_period_seconds,
            validate_id=False,
        )

        task_id = self._first_free_id()
        if task_id is None:
            raise TaskStoreError(f"task limit reached ({self.max_tasks})")

        record = TaskRecord(
            task_id=task_id,
            description=description,
            text=text,
            method=method,
            skills=skills,
            timer_period_seconds=timer_period_seconds,
            enabled=enabled,
        )
        self._tasks[task_id] = record
        try:
            self._save()
        except Exception:
            self._tasks.pop(task_id, None)
            raise
        return record

    def get(self, task_id: int) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def require(self, task_id: int) -> TaskRecord:
        task = self.get(task_id)
        if task is None:
            raise TaskStoreError(f"unknown task: {task_id}")
        return task

    def list(self) -> tuple[TaskRecord, ...]:
        return tuple(self._tasks[key] for key in sorted(self._tasks))

    def set_enabled(self, task_id: int, enabled: bool) -> TaskRecord:
        current = self.require(task_id)
        updated = replace(current, enabled=bool(enabled))
        self._replace(updated)
        return updated

    def set_timer_period(self, task_id: int, period_seconds: float) -> TaskRecord:
        if period_seconds <= 0:
            raise ValueError("period_seconds must be > 0")
        current = self.require(task_id)
        updated = replace(current, timer_period_seconds=float(period_seconds))
        self._replace(updated)
        return updated

    def delete(self, task_id: int) -> bool:
        record = self._tasks.pop(task_id, None)
        if record is None:
            return False
        try:
            self._save()
        except Exception:
            self._tasks[task_id] = record
            raise
        return True

    def status_text(self) -> str:
        tasks = self.list()
        if not tasks:
            return "no tasks"
        return "\n".join(
            f"TASK {task.task_id} {task.description}" for task in tasks
        )

    def _replace(self, record: TaskRecord) -> None:
        previous = self.require(record.task_id)
        self._tasks[record.task_id] = record
        try:
            self._save()
        except Exception:
            self._tasks[record.task_id] = previous
            raise

    def _validate_record(
        self,
        task_id: int,
        description: str,
        text: str,
        method: str,
        skills: tuple[str, ...],
        timer_period_seconds: float | None,
        *,
        line_number: int | None = None,
        validate_id: bool = True,
    ) -> None:
        where = f" at {self.path}:{line_number}" if line_number is not None else ""
        if validate_id and (task_id < 1 or task_id > self.max_tasks):
            raise TaskStoreError(
                f"task id {task_id}{where} is outside 1..{self.max_tasks}"
            )
        if not description:
            raise TaskStoreError(f"empty task description{where}")
        if not text:
            raise TaskStoreError(f"empty task text{where}")
        if method not in TASK_METHODS:
            raise TaskStoreError(f"invalid task method {method!r}{where}")
        if any(not name for name in skills):
            raise TaskStoreError(f"empty task skill name{where}")
        if len(set(skills)) != len(skills):
            raise TaskStoreError(f"duplicate task skill name{where}")
        if timer_period_seconds is not None and timer_period_seconds <= 0:
            raise TaskStoreError(f"timer_period_seconds must be > 0{where}")

    def _first_free_id(self) -> int | None:
        for task_id in range(1, self.max_tasks + 1):
            if task_id not in self._tasks:
                return task_id
        return None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(
                {
                    "task": task.task_id,
                    "description": task.description,
                    "text": task.text,
                    "method": task.method,
                    "skills": list(task.skills),
                    "timer_period_seconds": task.timer_period_seconds,
                    "enabled": task.enabled,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for task in self.list()
        )

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
