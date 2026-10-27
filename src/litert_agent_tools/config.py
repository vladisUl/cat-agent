"""Settings for native MQTT tools, independent of the existing CORE."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class MqttConfig:
    host: str = "192.168.0.21"
    port: int = 1883
    wait_seconds: int = 5

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("MQTT host must not be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("MQTT port must be between 1 and 65535")
        if not 1 <= self.wait_seconds <= 60:
            raise ValueError("MQTT wait_seconds must be between 1 and 60")

    @classmethod
    def from_env(cls) -> "MqttConfig":
        return cls(
            host=os.getenv("CAT_AGENT_MQTT_HOST", "192.168.0.21"),
            port=int(os.getenv("CAT_AGENT_MQTT_PORT", "1883")),
            wait_seconds=int(os.getenv("LITERT_TOOLS_MQTT_WAIT", "5")),
        )
