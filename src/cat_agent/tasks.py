from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile


DEFAULT_TASK_FILE = Path("/var/lib/cat-agent/task.txt")


class TaskStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: int
    description: str
    text: str


class TaskStore:
    """Small persistent registry for long-lived tasks.

    The file is deliberately independent from model/session state. A process can
    die, all KV state can disappear, and the task registry still survives.
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
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise TaskStoreError(
                    f"invalid task record at {self.path}:{line_number}"
                ) from exc

            if task_id < 1 or task_id > self.max_tasks:
                raise TaskStoreError(
                    f"task id {task_id} at {self.path}:{line_number} is outside 1..{self.max_tasks}"
                )
            if task_id in tasks:
                raise TaskStoreError(
                    f"duplicate task id {task_id} at {self.path}:{line_number}"
                )
            if not description:
                raise TaskStoreError(
                    f"empty task description at {self.path}:{line_number}"
                )
            if not text:
                raise TaskStoreError(f"empty task text at {self.path}:{line_number}")

            tasks[task_id] = TaskRecord(task_id, description, text)

        self._tasks = tasks

    def create(self, description: str, text: str) -> TaskRecord:
        description = description.strip()
        text = text.strip()
        if not description:
            raise ValueError("task description must not be empty")
        if not text:
            raise ValueError("task text must not be empty")

        task_id = self._first_free_id()
        if task_id is None:
            raise TaskStoreError(f"task limit reached ({self.max_tasks})")

        record = TaskRecord(task_id, description, text)
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
