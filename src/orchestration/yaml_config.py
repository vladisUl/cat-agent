from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "cat-agent.yaml"


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _check_keys(section: dict[str, Any], name: str, allowed: set[str]) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} option(s): {', '.join(sorted(unknown))}")


def _string(section: dict[str, Any], key: str, name: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value.strip()


def _bool(section: dict[str, Any], key: str, name: str) -> bool:
    value = section.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{name}.{key} must be a boolean")
    return value


def _int(
    section: dict[str, Any],
    key: str,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}.{key} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name}.{key} must be <= {maximum}")
    return value


def _number(
    section: dict[str, Any],
    key: str,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}.{key} must be a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name}.{key} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name}.{key} must be <= {maximum}")
    return result


@dataclass(frozen=True, slots=True)
class LiteRTProfile:
    model: str
    backend: str
    activation_dtype: str | None
    cpu_threads: int | None
    speculative: bool
    ynnpack: bool


@dataclass(frozen=True, slots=True)
class LiteRTConfig:
    active_profile: str
    profiles: dict[str, LiteRTProfile]

    @property
    def active(self) -> LiteRTProfile:
        return self.profiles[self.active_profile]


@dataclass(frozen=True, slots=True)
class OpenAIConfig:
    base_url: str
    model: str
    reasoning_effort: str


@dataclass(frozen=True, slots=True)
class AgentConfig:
    count: int
    manager_max_steps: int
    agent_max_steps: int
    manager_max_output_tokens: int
    agent_max_output_tokens: int
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    workspace: str
    prompt_dir: str
    command_timeout: int
    http_timeout: int
    request_retries: int
    retry_delay: float
    max_file_bytes: int


@dataclass(frozen=True, slots=True)
class WebConfig:
    host: str
    http_port: int
    ws_port: int


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    wake_words: tuple[str, ...]
    no_command_timeout: float
    max_command_seconds: float


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True, slots=True)
class NotificationsConfig:
    firebase_credentials: str
    firebase_tokens: str
    firebase_title: str
    ttl: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    litert: LiteRTConfig
    openai: OpenAIConfig
    agent: AgentConfig
    runtime: RuntimeConfig
    web: WebConfig
    voice: VoiceConfig
    logging: LoggingConfig
    notifications: NotificationsConfig


def _parse_litert(data: Any) -> LiteRTConfig:
    section = _mapping(data, "litert")
    _check_keys(section, "litert", {"active_profile", "profiles"})
    active_profile = _string(section, "active_profile", "litert")
    raw_profiles = _mapping(section.get("profiles"), "litert.profiles")
    if not raw_profiles:
        raise ValueError("litert.profiles must not be empty")

    profiles: dict[str, LiteRTProfile] = {}
    allowed = {
        "model",
        "backend",
        "activation_dtype",
        "cpu_threads",
        "speculative",
        "ynnpack",
    }
    for profile_name, raw in raw_profiles.items():
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise ValueError("litert.profiles keys must be non-empty strings")
        name = f"litert.profiles.{profile_name}"
        item = _mapping(raw, name)
        _check_keys(item, name, allowed)
        backend = _string(item, "backend", name).lower()
        if backend not in {"cpu", "gpu"}:
            raise ValueError(f"{name}.backend must be cpu or gpu")

        activation_dtype = item.get("activation_dtype")
        if activation_dtype is not None:
            if not isinstance(activation_dtype, str):
                raise ValueError(f"{name}.activation_dtype must be a string")
            activation_dtype = activation_dtype.strip().lower()
            if activation_dtype not in {"fp32", "fp16", "int16", "int8"}:
                raise ValueError(
                    f"{name}.activation_dtype must be one of fp32, fp16, int16, int8"
                )

        cpu_threads = item.get("cpu_threads")
        if cpu_threads is not None:
            if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
                raise ValueError(f"{name}.cpu_threads must be a positive integer")

        profiles[profile_name] = LiteRTProfile(
            model=_string(item, "model", name),
            backend=backend,
            activation_dtype=activation_dtype,
            cpu_threads=cpu_threads,
            speculative=_bool(item, "speculative", name),
            ynnpack=_bool(item, "ynnpack", name),
        )

    if active_profile not in profiles:
        raise ValueError(f"litert.active_profile references unknown profile {active_profile!r}")
    return LiteRTConfig(active_profile=active_profile, profiles=profiles)


def _parse_openai(data: Any) -> OpenAIConfig:
    section = _mapping(data, "openai")
    _check_keys(section, "openai", {"base_url", "model", "reasoning_effort"})
    reasoning = _string(section, "reasoning_effort", "openai").lower()
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    if reasoning not in allowed:
        raise ValueError(f"openai.reasoning_effort must be one of {sorted(allowed)}")
    return OpenAIConfig(
        base_url=_string(section, "base_url", "openai").rstrip("/"),
        model=_string(section, "model", "openai"),
        reasoning_effort=reasoning,
    )


def _parse_agent(data: Any) -> AgentConfig:
    section = _mapping(data, "agent")
    _check_keys(
        section,
        "agent",
        {
            "count",
            "manager_max_steps",
            "agent_max_steps",
            "manager_max_output_tokens",
            "agent_max_output_tokens",
            "temperature",
            "top_p",
        },
    )
    return AgentConfig(
        count=_int(section, "count", "agent", minimum=1),
        manager_max_steps=_int(section, "manager_max_steps", "agent", minimum=1),
        agent_max_steps=_int(section, "agent_max_steps", "agent", minimum=1),
        manager_max_output_tokens=_int(
            section, "manager_max_output_tokens", "agent", minimum=16
        ),
        agent_max_output_tokens=_int(
            section, "agent_max_output_tokens", "agent", minimum=16
        ),
        temperature=_number(section, "temperature", "agent", minimum=0.0, maximum=2.0),
        top_p=_number(section, "top_p", "agent", minimum=0.0, maximum=1.0),
    )


def _parse_runtime(data: Any) -> RuntimeConfig:
    section = _mapping(data, "runtime")
    _check_keys(
        section,
        "runtime",
        {
            "workspace",
            "prompt_dir",
            "command_timeout",
            "http_timeout",
            "request_retries",
            "retry_delay",
            "max_file_bytes",
        },
    )
    return RuntimeConfig(
        workspace=_string(section, "workspace", "runtime"),
        prompt_dir=_string(section, "prompt_dir", "runtime"),
        command_timeout=_int(section, "command_timeout", "runtime", minimum=1),
        http_timeout=_int(section, "http_timeout", "runtime", minimum=1),
        request_retries=_int(section, "request_retries", "runtime", minimum=0),
        retry_delay=_number(section, "retry_delay", "runtime", minimum=0.0),
        max_file_bytes=_int(section, "max_file_bytes", "runtime", minimum=1),
    )


def _parse_web(data: Any) -> WebConfig:
    section = _mapping(data, "web")
    _check_keys(section, "web", {"host", "http_port", "ws_port"})
    return WebConfig(
        host=_string(section, "host", "web"),
        http_port=_int(section, "http_port", "web", minimum=1, maximum=65535),
        ws_port=_int(section, "ws_port", "web", minimum=1, maximum=65535),
    )


def _parse_voice(data: Any) -> VoiceConfig:
    section = _mapping(data, "voice")
    _check_keys(section, "voice", {"wake_words", "no_command_timeout", "max_command_seconds"})
    raw_words = section.get("wake_words")
    if not isinstance(raw_words, list) or not raw_words:
        raise ValueError("voice.wake_words must be a non-empty list")
    words: list[str] = []
    for raw in raw_words:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("voice.wake_words entries must be non-empty strings")
        word = raw.strip().lower()
        if word in words:
            raise ValueError(f"voice.wake_words contains duplicate {word!r}")
        words.append(word)
    return VoiceConfig(
        wake_words=tuple(words),
        no_command_timeout=_number(section, "no_command_timeout", "voice", minimum=0.1),
        max_command_seconds=_number(section, "max_command_seconds", "voice", minimum=0.1),
    )


def _parse_logging(data: Any) -> LoggingConfig:
    section = _mapping(data, "logging")
    _check_keys(section, "logging", {"level"})
    level = _string(section, "level", "logging").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("logging.level must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
    return LoggingConfig(level=level)


def _parse_notifications(data: Any) -> NotificationsConfig:
    section = _mapping(data, "notifications")
    _check_keys(
        section,
        "notifications",
        {"firebase_credentials", "firebase_tokens", "firebase_title", "ttl"},
    )
    return NotificationsConfig(
        firebase_credentials=_string(section, "firebase_credentials", "notifications"),
        firebase_tokens=_string(section, "firebase_tokens", "notifications"),
        firebase_title=_string(section, "firebase_title", "notifications"),
        ttl=_int(section, "ttl", "notifications", minimum=0),
    )


def load_app_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = os.getenv("CAT_AGENT_CONFIG", "").strip() or DEFAULT_CONFIG_PATH
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cat-agent config not found: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    root = _mapping(raw, "cat-agent config")
    allowed = {"litert", "openai", "agent", "runtime", "web", "voice", "logging", "notifications"}
    _check_keys(root, "top-level", allowed)
    missing = allowed - set(root)
    if missing:
        raise ValueError(f"Missing cat-agent config section(s): {', '.join(sorted(missing))}")

    return AppConfig(
        litert=_parse_litert(root["litert"]),
        openai=_parse_openai(root["openai"]),
        agent=_parse_agent(root["agent"]),
        runtime=_parse_runtime(root["runtime"]),
        web=_parse_web(root["web"]),
        voice=_parse_voice(root["voice"]),
        logging=_parse_logging(root["logging"]),
        notifications=_parse_notifications(root["notifications"]),
    )
