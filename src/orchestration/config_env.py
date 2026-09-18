from __future__ import annotations

import json
import shlex
import sys

from .yaml_config import AppConfig, load_app_config


def _export(name: str, value: object) -> None:
    print(f"export {name}={shlex.quote(str(value))}")


def _unset(name: str) -> None:
    print(f"unset {name}")


def _export_common(config: AppConfig, *, reasoning_effort: str) -> None:
    agent = config.agent
    runtime = config.runtime
    notifications = config.notifications

    _export("CAT_AGENT_WORKSPACE", runtime.workspace)
    _export("CAT_AGENT_PROMPT_DIR", runtime.prompt_dir)
    _export("CAT_AGENT_AGENT_COUNT", agent.count)
    _export("CAT_AGENT_MAX_MANAGER_STEPS", agent.manager_max_steps)
    _export("CAT_AGENT_MAX_AGENT_STEPS", agent.agent_max_steps)
    _export("CAT_AGENT_MAX_FILE_BYTES", runtime.max_file_bytes)
    _export("CAT_AGENT_COMMAND_TIMEOUT_SECONDS", runtime.command_timeout)
    _export("CAT_AGENT_HTTP_TIMEOUT_SECONDS", runtime.http_timeout)
    _export("CAT_AGENT_REQUEST_RETRIES", runtime.request_retries)
    _export("CAT_AGENT_RETRY_DELAY_SECONDS", runtime.retry_delay)
    _export("CAT_AGENT_MANAGER_MAX_OUTPUT_TOKENS", agent.manager_max_output_tokens)
    _export("CAT_AGENT_AGENT_MAX_OUTPUT_TOKENS", agent.agent_max_output_tokens)
    _export("CAT_AGENT_TEMPERATURE", agent.temperature)
    _export("CAT_AGENT_TOP_P", agent.top_p)
    _export("CAT_AGENT_REASONING_EFFORT", reasoning_effort)
    _export("CAT_AGENT_LOG_LEVEL", config.logging.level)

    _export("CAT_AGENT_FIREBASE_CREDENTIALS", notifications.firebase_credentials)
    _export("CAT_AGENT_FIREBASE_TOKENS", notifications.firebase_tokens)
    _export("CAT_AGENT_FIREBASE_TITLE", notifications.firebase_title)
    _export("CAT_AGENT_FIREBASE_TTL", notifications.ttl)


def _export_litert(config: AppConfig) -> None:
    _export_common(config, reasoning_effort="none")
    profile = config.litert.active
    _export("LITERT_AGENT_MODEL_PATH", profile.model)
    _export("LITERT_AGENT_BACKEND", profile.backend)
    _export("LITERT_AGENT_SPECULATIVE", "1" if profile.speculative else "0")
    _export("LITERT_AGENT_YNNPACK", "1" if profile.ynnpack else "0")
    if profile.cpu_threads is None:
        _unset("LITERT_AGENT_CPU_THREADS")
    else:
        _export("LITERT_AGENT_CPU_THREADS", profile.cpu_threads)
    if profile.activation_dtype is None:
        _unset("LITERT_AGENT_ACTIVATION_DATA_TYPE")
    else:
        _export("LITERT_AGENT_ACTIVATION_DATA_TYPE", profile.activation_dtype)


def _export_openai(config: AppConfig) -> None:
    profile = config.openai.active
    _export_common(config, reasoning_effort=profile.reasoning_effort)
    _export("CAT_AGENT_API_BASE_URL", profile.base_url)
    _export("CAT_AGENT_MODEL", profile.model)
    _export("CAT_AGENT_OPENAI_READINESS", profile.readiness)


def _export_web(config: AppConfig) -> None:
    _export("CAT_AGENT_WEB_HOST", config.web.host)
    _export("CAT_AGENT_HTTP_PORT", config.web.http_port)
    _export("CAT_AGENT_WS_PORT", config.web.ws_port)


def _export_voice(config: AppConfig) -> None:
    _export(
        "CAT_AGENT_WAKE_WORDS_JSON",
        json.dumps(config.voice.wake_words, ensure_ascii=False),
    )
    _export("CAT_AGENT_NO_COMMAND_TIMEOUT_SECONDS", config.voice.no_command_timeout)
    _export("CAT_AGENT_MAX_COMMAND_SECONDS", config.voice.max_command_seconds)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in {"litert", "openai", "web", "voice"}:
        print("Usage: python -m orchestration.config_env litert|openai|web|voice", file=sys.stderr)
        return 2

    config = load_app_config()
    target = args[0]
    if target == "litert":
        _export_litert(config)
    elif target == "openai":
        _export_openai(config)
    elif target == "web":
        _export_web(config)
    else:
        _export_voice(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
