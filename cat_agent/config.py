from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    workspace: Path
    prompt_dir: Path
    api_base_url: str
    model: str
    agent_count: int
    max_manager_steps: int
    max_agent_steps: int
    max_file_bytes: int
    command_timeout_seconds: int
    http_timeout_seconds: int
    request_retries: int
    retry_delay_seconds: float
    manager_max_output_tokens: int
    agent_max_output_tokens: int
    temperature: float
    top_p: float
    reasoning_effort: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent.parent
        model = os.getenv("CAT_AGENT_MODEL", "").strip()
        if not model:
            raise ValueError("CAT_AGENT_MODEL must be set")

        reasoning_effort = os.getenv("CAT_AGENT_REASONING_EFFORT", "none").strip().lower()
        allowed_reasoning = {"none", "minimal", "low", "medium", "high", "xhigh"}
        if reasoning_effort not in allowed_reasoning:
            raise ValueError(
                "CAT_AGENT_REASONING_EFFORT must be one of "
                f"{sorted(allowed_reasoning)}, got {reasoning_effort!r}"
            )

        return cls(
            workspace=Path(os.getenv("CAT_AGENT_WORKSPACE", "/opt/model")),
            prompt_dir=Path(os.getenv("CAT_AGENT_PROMPT_DIR", str(project_root / "prompts"))),
            api_base_url=os.getenv(
                "CAT_AGENT_API_BASE_URL", "http://127.0.0.1:9380/v1"
            ).rstrip("/"),
            model=model,
            agent_count=_env_int("CAT_AGENT_AGENT_COUNT", 3, minimum=1),
            max_manager_steps=_env_int("CAT_AGENT_MAX_MANAGER_STEPS", 12, minimum=1),
            max_agent_steps=_env_int("CAT_AGENT_MAX_AGENT_STEPS", 12, minimum=1),
            max_file_bytes=_env_int("CAT_AGENT_MAX_FILE_BYTES", 65536, minimum=1),
            command_timeout_seconds=_env_int(
                "CAT_AGENT_COMMAND_TIMEOUT_SECONDS", 20, minimum=1
            ),
            http_timeout_seconds=_env_int("CAT_AGENT_HTTP_TIMEOUT_SECONDS", 300, minimum=1),
            request_retries=_env_int("CAT_AGENT_REQUEST_RETRIES", 0, minimum=0),
            retry_delay_seconds=_env_float(
                "CAT_AGENT_RETRY_DELAY_SECONDS", 2.0, minimum=0.0
            ),
            manager_max_output_tokens=_env_int(
                "CAT_AGENT_MANAGER_MAX_OUTPUT_TOKENS", 256, minimum=16
            ),
            agent_max_output_tokens=_env_int(
                "CAT_AGENT_AGENT_MAX_OUTPUT_TOKENS", 128, minimum=16
            ),
            temperature=_env_float(
                "CAT_AGENT_TEMPERATURE", 0.0, minimum=0.0, maximum=2.0
            ),
            top_p=_env_float("CAT_AGENT_TOP_P", 1.0, minimum=0.0, maximum=1.0),
            reasoning_effort=reasoning_effort,
            log_level=os.getenv("CAT_AGENT_LOG_LEVEL", "INFO").upper(),
        )
