"""Create native MQTT callables from JSON: no built-in topic/function registry."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
import subprocess
from typing import Callable

LOGGER = logging.getLogger(__name__)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _load(path: Path) -> dict:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load MQTT JSON {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return doc


def _int_range(value, low: int, high: int, label: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")
    return value


def _definitions(directory: Path) -> list[tuple]:
    settings = _load(directory / "mqtt.json")
    if set(settings) != {"port", "timeout", "servers"}:
        raise ValueError("mqtt.json requires port, timeout, servers")
    port = _int_range(settings["port"], 1, 65535, "MQTT port")
    timeout = _int_range(settings["timeout"], 1, 60, "MQTT timeout")
    servers = settings["servers"]
    if not isinstance(servers, dict) or not servers:
        raise ValueError("MQTT servers must be a nonempty object")
    brokers = {}
    for alias, address in servers.items():
        if not isinstance(alias, str) or not alias.strip():
            raise ValueError("MQTT server alias must be nonempty")
        if isinstance(address, str):
            host, broker_port = address, port
        elif isinstance(address, dict) and set(address) == {"host", "port"}:
            host = address["host"]
            broker_port = _int_range(address["port"], 1, 65535,
                                     f"MQTT server {alias} port")
        else:
            raise ValueError(f"MQTT server {alias}: expected host or host/port")
        if not isinstance(host, str) or not host.strip() or "\x00" in host:
            raise ValueError(f"MQTT server {alias}: invalid host")
        brokers[alias] = (host, broker_port, timeout)

    raw = _load(directory / "mqtt_tools.json")
    if set(raw) != {"tools"} or not isinstance(raw["tools"], list):
        raise ValueError("mqtt_tools.json requires a tools array")
    definitions = []
    names = set()
    for index, item in enumerate(raw["tools"]):
        label = f"mqtt_tools.json tools[{index}]"
        mandatory = {"name", "description", "server", "topic", "fields"}
        if not isinstance(item, dict) or not (
            mandatory <= set(item) <= mandatory | {"validation"}
        ):
            raise ValueError(f"{label}: expected name/description/server/topic/fields")
        name, description = item["name"], item["description"]
        server, topic, fields = item["server"], item["topic"], item["fields"]
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in names:
            raise ValueError(f"{label}: invalid or duplicate tool name {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label}: description must be nonempty")
        if not isinstance(server, str) or server not in brokers:
            raise ValueError(f"{label}: unknown server {server!r}")
        if not isinstance(topic, str) or not topic.strip() or any(c in topic for c in "\x00+#"):
            raise ValueError(f"{label}: topic must be an explicit single MQTT topic")
        if (not isinstance(fields, dict) or not fields or
                not all(isinstance(k, str) and k and isinstance(v, str) and v
                        for k, v in fields.items()) or
                len(set(fields.values())) != len(fields) or "ok" in fields.values()):
            raise ValueError(f"{label}: fields must map source fields to unique output keys")
        checks = item.get("validation", {})
        if not isinstance(checks, dict) or not set(checks).issubset(fields):
            raise ValueError(f"{label}: validation refers to unknown fields")
        for field, check in checks.items():
            if not isinstance(check, dict) or check.get("type") not in (
                "number", "string", "boolean"
            ) or set(check) - {"type", "min", "max"}:
                raise ValueError(f"{label}: invalid validation rule for {field}")
            if check["type"] != "number" and set(check) != {"type"}:
                raise ValueError(f"{label}: min/max only valid for numbers")
            for bound in ("min", "max"):
                if bound in check and (type(check[bound]) not in (int, float)
                                       or not math.isfinite(check[bound])):
                    raise ValueError(f"{label}: non-finite {bound} for {field}")
            if "min" in check and "max" in check and check["min"] > check["max"]:
                raise ValueError(f"{label}: min > max for {field}")
        names.add(name)
        definitions.append((name, description, brokers[server], topic, dict(fields),
                            {k: dict(v) for k, v in checks.items()}))
    return definitions


def _read_measurement(broker: tuple, topic: str, fields: dict, validation: dict) -> dict:
    host, port, timeout = broker
    command = ["mosquitto_sub", "-h", host, "-p", str(port), "-t", topic,
               "-C", "1", "-W", str(timeout)]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=timeout + 2, check=False)
    except FileNotFoundError:
        return {"ok": False, "error": "mosquitto_sub is not installed"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "MQTT read timed out"}
    except OSError as exc:
        return {"ok": False, "error": f"MQTT process error: {exc}"}
    if result.returncode != 0:
        return {"ok": False, "error": f"MQTT read failed (exit {result.returncode}): {result.stderr.strip()[:300]}"}
    try:
        payload = json.loads(result.stdout)
        if not isinstance(payload, dict):
            raise ValueError("MQTT payload must be a JSON object")
        output = {"ok": True}
        for source, destination in fields.items():
            value = payload[source]
            rule = validation.get(source, {})
            kind = rule.get("type")
            if kind == "number":
                if type(value) not in (int, float):
                    raise ValueError(f"{source} must be a JSON number")
                value = float(value)
            elif kind == "string" and not isinstance(value, str):
                raise ValueError(f"{source} must be a JSON string")
            elif kind == "boolean" and type(value) is not bool:
                raise ValueError(f"{source} must be a JSON boolean")
            if type(value) is float and not math.isfinite(value):
                raise ValueError(f"{source} is not finite")
            if kind == "number":
                if "min" in rule and value < rule["min"]:
                    raise ValueError(f"{source} below minimum")
                if "max" in rule and value > rule["max"]:
                    raise ValueError(f"{source} above maximum")
            output[destination] = value
        return output
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return {"ok": False, "error": f"Invalid MQTT payload: {exc}"}


def _make_tool(name: str, description: str, broker: tuple, topic: str,
               fields: dict, validation: dict) -> Callable[[], dict]:
    def tool() -> dict:
        LOGGER.info("native-tool CALL name=%s topic=%s", name, topic)
        result = _read_measurement(broker, topic, fields, validation)
        LOGGER.info("native-tool RESULT name=%s result=%s",
                    name, json.dumps(result, ensure_ascii=False))
        return result

    tool.__name__ = name
    tool.__qualname__ = name
    tool.__doc__ = description
    return tool


def create_mqtt_tools(config_dir: Path | None = None) -> list[Callable[[], dict]]:
    """Read JSON and return regular Python functions for Conversation(tools=...)."""
    directory = Path(config_dir) if config_dir is not None else CONFIG_DIR
    tools = [_make_tool(*values) for values in _definitions(directory)]
    LOGGER.info("native-tools MQTT loaded=%s from=%s",
                tuple(fn.__name__ for fn in tools), directory)
    return tools
