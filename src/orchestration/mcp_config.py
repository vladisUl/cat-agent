"""Optional MCP configuration; importing this module does not import the SDK."""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class McpServerConfig:
    name: str
    enabled: bool
    transport: str
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    connect_timeout_seconds: float = 10.0
    call_timeout_seconds: float = 30.0
    reconnect_delay_seconds: float = 5.0
    env: tuple[tuple[str, str], ...] = ()
    headers_env: tuple[tuple[str, str], ...] = ()
    manager: bool = True
    description: str = ""


def parse_mcp_config(raw: object) -> tuple[McpServerConfig, ...]:
    if not isinstance(raw, dict) or set(raw) - {"servers"}:
        raise ValueError("mcp must be a mapping containing only servers")
    servers = raw.get("servers", [])
    if not isinstance(servers, list):
        raise ValueError("mcp.servers must be a list")
    result = []
    names = set()
    allowed = set(McpServerConfig.__dataclass_fields__)
    for item in servers:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError("Invalid mcp server fields")
        name = item.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", name):
            raise ValueError("mcp server name must match [a-z][a-z0-9_-]*")
        if name in names:
            raise ValueError(f"Duplicate mcp server: {name}")
        names.add(name)
        if not isinstance(item.get("enabled"), bool):
            raise ValueError(f"mcp {name}: enabled must be boolean")
        transport = item.get("transport")
        if transport not in {"stdio", "streamable_http"}:
            raise ValueError(f"mcp {name}: transport must be stdio or streamable_http")
        values = dict(item)
        manager = item.get("manager", True)
        if not isinstance(manager, bool):
            raise ValueError(f"mcp {name}: manager must be boolean")
        values["manager"] = manager
        description = item.get("description", "")
        if not isinstance(description, str) or "\n" in description or "\r" in description:
            raise ValueError(f"mcp {name}: description must be a single-line string")
        values["description"] = description.strip()
        if transport == "stdio":
            if not isinstance(item.get("command"), str) or not item["command"].strip():
                raise ValueError(f"mcp {name}: command is required")
            args = item.get("args")
            if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
                raise ValueError(f"mcp {name}: args must be a list of strings")
            values["args"] = tuple(args)
            if "url" in item or "headers_env" in item:
                raise ValueError(f"mcp {name}: HTTP settings used with stdio")
        else:
            url = item.get("url")
            if not isinstance(url, str):
                raise ValueError(f"mcp {name}: url is required")
            try:
                parsed = urlsplit(url)
                valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                         and not parsed.username and not parsed.password and not parsed.fragment)
                parsed.port
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(f"mcp {name}: invalid HTTP URL")
            if any(key in item for key in ("command", "args", "env")):
                raise ValueError(f"mcp {name}: stdio settings used with HTTP")
        for field in ("connect_timeout_seconds", "call_timeout_seconds", "reconnect_delay_seconds"):
            if field in item:
                value = item[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"mcp {name}: {field} must be finite and positive")
        for field in ("env", "headers_env"):
            if field not in item:
                continue
            value = item[field]
            if not isinstance(value, dict) or any(
                not isinstance(k, str) or not k or '\n' in k or '\r' in k
                or not isinstance(v, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v)
                for k, v in value.items()
            ):
                raise ValueError(f"mcp {name}: {field} must map names to environment variable names")
            values[field] = tuple(sorted(value.items()))
        result.append(McpServerConfig(**values))
    return tuple(result)


def load_mcp_config() -> tuple[McpServerConfig, ...]:
    """Read only the optional section, keeping environment-only launch working."""
    import yaml
    from .yaml_config import DEFAULT_CONFIG_PATH

    explicit = os.getenv("CAT_AGENT_CONFIG", "").strip()
    path = Path(explicit) if explicit else DEFAULT_CONFIG_PATH
    if not explicit and not path.exists():
        return ()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("cat-agent config must be a mapping")
    return parse_mcp_config(raw.get("mcp", {}))
