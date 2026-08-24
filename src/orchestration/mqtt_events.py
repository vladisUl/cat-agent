from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import threading

from .event_store import EventBinding, EventStore


LOGGER = logging.getLogger(__name__)
MQTT_HOST = os.getenv("CAT_AGENT_MQTT_HOST", "192.168.0.21").strip() or "192.168.0.21"
MQTT_PORT = int(os.getenv("CAT_AGENT_MQTT_PORT", "1883"))
_FIELD_RULE_RE = re.compile(
    r"^(?P<topic>[^:\s]+):.*;\s*"
    r"(?P<field>[A-Za-z_][A-Za-z0-9_]*):\s*"
    r"(?P<value_type>[A-Za-z_][A-Za-z0-9_]*)\s*;\s*"
    r"\((?P<values>.+)\)\s*$"
)


@dataclass(frozen=True, slots=True)
class MqttFieldRule:
    topic: str
    field: str
    value_type: str
    values: tuple[str, ...]


class MqttTopicCatalog:
    """Read discrete MQTT field semantics from mqtt.txt.

    A discrete line has the form:
      topic: description; field: type; (value: meaning, value: meaning)

    The text after each value is for the models. SYSTEM only needs the listed
    values to know which state transitions must wake the saved TASK/QUERY.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._rules: dict[tuple[str, str], MqttFieldRule] = {}
        self.reload()

    def reload(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(f"MQTT catalog not found: {self.path}")

        rules: dict[tuple[str, str], MqttFieldRule] = {}
        for line_number, raw in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw.strip()
            if not line or line == "topics:":
                continue
            match = _FIELD_RULE_RE.fullmatch(line)
            if match is None:
                continue

            topic = match.group("topic").strip()
            field = match.group("field").strip()
            value_type = match.group("value_type").strip().lower()
            values: list[str] = []
            for item in match.group("values").split(","):
                token, separator, _meaning = item.partition(":")
                if not separator:
                    raise ValueError(
                        f"invalid discrete MQTT value at {self.path}:{line_number}"
                    )
                value = token.strip()
                if value_type == "boolean":
                    value = value.lower()
                    if value not in {"true", "false"}:
                        raise ValueError(
                            f"boolean MQTT value must be true or false at "
                            f"{self.path}:{line_number}"
                        )
                if not value:
                    raise ValueError(
                        f"empty discrete MQTT value at {self.path}:{line_number}"
                    )
                values.append(value)

            if not values or len(set(values)) != len(values):
                raise ValueError(
                    f"invalid discrete MQTT values at {self.path}:{line_number}"
                )
            rule = MqttFieldRule(topic, field, value_type, tuple(values))
            rules[(topic, field)] = rule

        self._rules = rules

    def require(self, topic: str, field: str) -> MqttFieldRule:
        key = (topic.strip(), field.strip())
        rule = self._rules.get(key)
        if rule is None:
            raise ValueError(
                f"MQTT field {key[0]} {key[1]} has no discrete value rule in mqtt.txt"
            )
        return rule


MqttEventCallback = Callable[[EventBinding, str], None]


class MqttEventMonitor:
    """Continuously receive MQTT payloads and emit configured discrete changes."""

    def __init__(
        self,
        event_store: EventStore,
        callback: MqttEventCallback,
        *,
        host: str = MQTT_HOST,
        port: int = MQTT_PORT,
    ) -> None:
        self.event_store = event_store
        self.callback = callback
        self.host = host
        self.port = port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[str] | None = None
        self._last_payloads: dict[str, dict[str, object]] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="cat-agent-mqtt-events",
            daemon=True,
        )
        self._thread.start()
        LOGGER.debug("MQTT event monitor started")

    def close(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        self._thread = None
        self._process = None
        LOGGER.debug("MQTT event monitor stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                process = subprocess.Popen(
                    [
                        "mosquitto_sub",
                        "-h",
                        self.host,
                        "-p",
                        str(self.port),
                        "-t",
                        "#",
                        "-v",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except OSError as exc:
                LOGGER.warning("MQTT event monitor cannot start mosquitto_sub: %s", exc)
                self._stop.wait(5.0)
                continue

            self._process = process
            stdout = process.stdout
            if stdout is None:
                process.terminate()
                self._stop.wait(1.0)
                continue

            try:
                for raw in stdout:
                    if self._stop.is_set():
                        break
                    self._consume_line(raw)
            finally:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
                self._process = None

            if not self._stop.is_set():
                LOGGER.warning("MQTT event monitor lost broker subscription; reconnecting")
                self._stop.wait(1.0)

    def _consume_line(self, raw: str) -> None:
        line = raw.strip()
        topic, separator, payload_text = line.partition(" ")
        if not separator or not topic or not payload_text:
            return
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return

        previous_payload = self._last_payloads.get(topic)
        self._last_payloads[topic] = payload

        for binding in self.event_store.snapshot():
            if binding.source != "mqtt" or binding.topic != topic:
                continue
            if binding.field not in payload:
                continue

            current = self._canonical_value(payload[binding.field], binding.value_type)
            if current is None:
                continue

            if binding.value_type == "boolean":
                if previous_payload is None or binding.field not in previous_payload:
                    continue
                previous = self._canonical_value(
                    previous_payload[binding.field],
                    binding.value_type,
                )
                if previous is None or previous == current:
                    continue
            if current not in binding.values:
                continue

            try:
                self.callback(binding, current)
            except Exception:
                LOGGER.exception(
                    "MQTT event callback failed task=%d topic=%s field=%s value=%s",
                    binding.task_id,
                    binding.topic,
                    binding.field,
                    current,
                )

    @staticmethod
    def _canonical_value(value: object, value_type: str) -> str | None:
        if value_type == "boolean":
            if value is True:
                return "true"
            if value is False:
                return "false"
            return None
        if isinstance(value, str):
            return value
        if value is None:
            return "null"
        if isinstance(value, (int, float)):
            return json.dumps(value, ensure_ascii=False)
        return None
