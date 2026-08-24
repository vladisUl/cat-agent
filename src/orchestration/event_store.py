from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import threading


DEFAULT_EVENT_FILE = Path("/var/lib/cat-agent/events.json")
_EVENT_NAME_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EventStoreError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EventBinding:
    name: str
    source: str
    task_id: int
    description: str
    topic: str = ""
    field: str = ""
    value_type: str = ""
    values: tuple[str, ...] = ()
    command: str = ""


class EventStore:
    """Persistent mapping from an external event name to a saved TASK/QUERY."""

    def __init__(self, path: Path = DEFAULT_EVENT_FILE) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._bindings: dict[str, EventBinding] = {}
        self.reload()

    def reload(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._bindings = {}
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise EventStoreError(f"cannot read event store {self.path}: {exc}") from exc
            if not isinstance(raw, dict):
                raise EventStoreError(f"invalid event store {self.path}: root must be an object")

            bindings: dict[str, EventBinding] = {}
            try:
                for name, item in raw.items():
                    if not isinstance(item, dict):
                        raise TypeError(f"event {name!r} must be an object")
                    raw_values = item.get("values", [])
                    if not isinstance(raw_values, list):
                        raise TypeError(f"event {name!r} values must be a list")
                    binding = self._validated_binding(
                        str(name),
                        str(item["source"]),
                        int(item["task_id"]),
                        str(item["description"]),
                        topic=str(item.get("topic", "")),
                        field=str(item.get("field", "")),
                        value_type=str(item.get("value_type", "")),
                        values=tuple(str(value) for value in raw_values),
                        command=str(item.get("command", "")),
                    )
                    bindings[binding.name] = binding
            except (KeyError, TypeError, ValueError) as exc:
                raise EventStoreError(f"invalid event store {self.path}: {exc}") from exc
            self._bindings = bindings

    def register(
        self,
        task_id: int,
        description: str,
        *,
        source: str = "gpio",
        name: str | None = None,
        topic: str = "",
        field: str = "",
        value_type: str = "",
        values: tuple[str, ...] = (),
        command: str = "",
    ) -> EventBinding:
        source = source.strip().lower()
        event_name = (name or f"task_{source}{task_id}").strip()
        binding = self._validated_binding(
            event_name,
            source,
            task_id,
            description,
            topic=topic,
            field=field,
            value_type=value_type,
            values=values,
            command=command,
        )
        with self._lock:
            current = self._bindings.get(binding.name)
            if current is not None and current.task_id != binding.task_id:
                raise EventStoreError(
                    f"event {binding.name!r} is already bound to task {current.task_id}"
                )
            previous = self._bindings.get(binding.name)
            self._bindings[binding.name] = binding
            try:
                self._save_locked()
            except Exception:
                if previous is None:
                    self._bindings.pop(binding.name, None)
                else:
                    self._bindings[binding.name] = previous
                raise
        return binding

    def resolve(self, source: str, name: str) -> EventBinding | None:
        source = source.strip().lower()
        name = name.strip()
        with self._lock:
            binding = self._bindings.get(name)
            if binding is None or binding.source != source:
                return None
            return binding

    def unregister_task(self, task_id: int) -> None:
        with self._lock:
            removed = {
                name: binding
                for name, binding in self._bindings.items()
                if binding.task_id == task_id
            }
            if not removed:
                return
            for name in removed:
                self._bindings.pop(name, None)
            try:
                self._save_locked()
            except Exception:
                self._bindings.update(removed)
                raise

    def snapshot(self) -> tuple[EventBinding, ...]:
        with self._lock:
            return tuple(self._bindings[name] for name in sorted(self._bindings))

    @staticmethod
    def _validated_binding(
        name: str,
        source: str,
        task_id: int,
        description: str,
        *,
        topic: str = "",
        field: str = "",
        value_type: str = "",
        values: tuple[str, ...] = (),
        command: str = "",
    ) -> EventBinding:
        name = name.strip()
        source = source.strip().lower()
        description = description.strip()
        topic = topic.strip()
        field = field.strip()
        value_type = value_type.strip().lower()
        values = tuple(value.strip() for value in values)
        command = command.strip()

        if not name or _EVENT_NAME_RE.fullmatch(name) is None:
            raise EventStoreError(f"invalid event name: {name!r}")
        if not source or _EVENT_NAME_RE.fullmatch(source) is None:
            raise EventStoreError(f"invalid event source: {source!r}")
        if task_id <= 0:
            raise EventStoreError("event task_id must be > 0")
        if not description:
            raise EventStoreError("event description must be non-empty")

        if source == "mqtt":
            if not topic:
                raise EventStoreError("mqtt event topic must be non-empty")
            if not field or _FIELD_RE.fullmatch(field) is None:
                raise EventStoreError(f"invalid mqtt event field: {field!r}")
            if not value_type:
                raise EventStoreError("mqtt event value_type must be non-empty")
            if not values or any(not value for value in values):
                raise EventStoreError("mqtt event values must be non-empty")
            if len(set(values)) != len(values):
                raise EventStoreError("mqtt event values must be unique")
            if not command:
                raise EventStoreError("mqtt event command must be non-empty")

        return EventBinding(
            name,
            source,
            task_id,
            description,
            topic,
            field,
            value_type,
            values,
            command,
        )

    def _save_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, dict[str, object]] = {}
        for binding in self.snapshot():
            item: dict[str, object] = {
                "source": binding.source,
                "task_id": binding.task_id,
                "description": binding.description,
            }
            if binding.topic:
                item["topic"] = binding.topic
            if binding.field:
                item["field"] = binding.field
            if binding.value_type:
                item["value_type"] = binding.value_type
            if binding.values:
                item["values"] = list(binding.values)
            if binding.command:
                item["command"] = binding.command
            payload[binding.name] = item

        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
