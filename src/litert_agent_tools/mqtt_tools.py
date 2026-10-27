"""Model-native, read-only MQTT tools for the four existing temperature sensors."""

from __future__ import annotations

import json
import logging
import math
import subprocess
from typing import Callable

from .config import MqttConfig


LOGGER = logging.getLogger(__name__)


TOPICS = {
    "outdoor": ("zigbee2mqtt/temp_ulica", ("temperature", "humidity")),
    "home": ("zigbee2mqtt/temp_tua", ("temperature", "humidity")),
    "aquarium": ("zigbee2mqtt/temp_akva", ("temperature",)),
    "freezer": ("zigbee2mqtt/temp_moroz", ("temperature",)),
}


def _read_measurement(config: MqttConfig, topic: str, fields: tuple[str, ...]) -> dict:
    """Read exactly one MQTT message; do not run a shell or use model-supplied topics."""
    command = [
        "mosquitto_sub", "-h", config.host, "-p", str(config.port),
        "-t", topic, "-C", "1", "-W", str(config.wait_seconds),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True,
            timeout=config.wait_seconds + 2, check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "mosquitto_sub is not installed"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "MQTT read timed out"}
    except OSError as exc:
        return {"ok": False, "error": f"MQTT process error: {exc}"}

    if result.returncode != 0:
        details = result.stderr.strip()[:300]
        return {"ok": False, "error": f"MQTT read failed (exit {result.returncode}): {details}"}

    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("MQTT payload must be a JSON object")
        values: dict[str, float] = {}
        for field in fields:
            raw = payload[field]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"{field} must be a JSON number")
            number = float(raw)
            if not math.isfinite(number):
                raise ValueError(f"{field} is not finite")
            if field == "humidity" and not 0 <= number <= 100:
                raise ValueError("humidity must be between 0 and 100 percent")
            values["temperature_c" if field == "temperature" else "humidity_percent"] = number
        return {"ok": True, **values}
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {"ok": False, "error": f"Invalid MQTT payload: {exc}"}



def _invoke_tool(name: str, config: MqttConfig, topic: str, fields: tuple[str, ...]) -> dict:
    """Log actual invocation and measured result, not the model's text."""
    LOGGER.info("native-tool CALL name=%s topic=%s", name, topic)
    result = _read_measurement(config, topic, fields)
    LOGGER.info("native-tool RESULT name=%s result=%s", name, json.dumps(result, ensure_ascii=False))
    return result


def create_mqtt_tools(config: MqttConfig | None = None) -> list[Callable[[], dict]]:
    """Return four zero-argument functions with stable names and model-facing docstrings."""
    settings = config if config is not None else MqttConfig.from_env()

    def get_outdoor_weather() -> dict:
        """Получить текущие температуру на улице (°C) и влажность воздуха (%)."""
        topic, fields = TOPICS["outdoor"]
        return _invoke_tool("get_outdoor_weather", settings, topic, fields)

    def get_home_climate() -> dict:
        """Получить текущие температуру дома (°C) и влажность воздуха (%)."""
        topic, fields = TOPICS["home"]
        return _invoke_tool("get_home_climate", settings, topic, fields)

    def get_aquarium_temperature() -> dict:
        """Получить текущую температуру воды в аквариуме (°C)."""
        topic, fields = TOPICS["aquarium"]
        return _invoke_tool("get_aquarium_temperature", settings, topic, fields)

    def get_freezer_temperature() -> dict:
        """Получить текущую температуру в морозильнике (°C)."""
        topic, fields = TOPICS["freezer"]
        return _invoke_tool("get_freezer_temperature", settings, topic, fields)

    return [
        get_outdoor_weather, get_home_climate,
        get_aquarium_temperature, get_freezer_temperature,
    ]
