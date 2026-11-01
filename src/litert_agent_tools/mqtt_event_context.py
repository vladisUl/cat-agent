"""Event-time value scoped to one synchronous native AGENT execution."""
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class MqttEventSnapshot:
    host: str
    port: int
    topic: str
    field: str
    value: object
    event_name: str


MQTT_EVENT_SNAPSHOT: ContextVar[MqttEventSnapshot | None] = ContextVar(
    'native_mqtt_event_snapshot', default=None,
)
