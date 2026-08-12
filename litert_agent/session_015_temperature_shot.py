from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import logging
import os
from pathlib import Path
import sys
import time

import litert_lm

from cat_agent.agent import AgentWorker
from cat_agent.config import Settings
from cat_agent.manager import ManagerRuntime
from cat_agent.pool import AgentPool
from cat_agent.prompt_store import AGENT_BOOTSTRAP_ACK, MANAGER_BOOTSTRAP_ACK, PromptStore
from cat_agent.skills import SkillBase

from .session_015_model_client import LiteRT015ChatClient, WarmResult

TASK = "Получить текущую температуру на улице."
DEFAULT_E4B_MODEL = Path(
    "/opt/litert-lm/models/gemma-4-E4B-it/gemma-4-E4B-it.litertlm"
)
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProbeBundle:
    runtime: ManagerRuntime
    manager_engine: litert_lm.Engine
    agent_engine: litert_lm.Engine
    manager_client: LiteRT015ChatClient
    agent_client: LiteRT015ChatClient
    manager_engine_init_seconds: float
    agent_engine_init_seconds: float

    def close(self) -> None:
        self.manager_client.close()
        self.agent_client.close()
        self.manager_engine.close()
        self.agent_engine.close()


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bundle = _build_bundle(settings)
    try:
        manager_messages = bundle.runtime.messages[:3]
        if (
            len(manager_messages) != 3
            or manager_messages[-1].get("role") != "assistant"
            or manager_messages[-1].get("content") != MANAGER_BOOTSTRAP_ACK
        ):
            raise RuntimeError("Unexpected canonical manager bootstrap history")
        manager_warm = bundle.manager_client.prepare_prefix(manager_messages)

        skills = bundle.runtime.skill_base.require(("mqtt",))
        bootstrap = bundle.runtime.prompt_store.build_agent_bootstrap(
            skills, settings.workspace
        )
        agent_messages = [
            {
                "role": "system",
                "content": bundle.runtime.prompt_store.agent_system_prompt("agent1"),
            },
            {"role": "user", "content": bootstrap},
            {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
        ]
        agent_warm = bundle.agent_client.prepare_prefix(agent_messages)

        started = time.monotonic()
        turn = bundle.runtime.user_message(TASK)
        chain_total = time.monotonic() - started

        print(f"CONTROL_RUNTIME_VERSION={_runtime_version()}")
        print(f"CONTROL_TASK={TASK}")
        print(f"CONTROL_RESULT_KIND={turn.kind}")
        print(f"CONTROL_RESULT={turn.text}")
        print(f"CONTROL_WARM_MANAGER_STRATEGY={manager_warm.strategy}")
        print(f"CONTROL_WARM_MANAGER={manager_warm.elapsed_seconds:.3f}s")
        print(f"CONTROL_WARM_MANAGER_TOKENS={manager_warm.token_count}")
        print(f"CONTROL_WARM_AGENT_STRATEGY={agent_warm.strategy}")
        print(f"CONTROL_WARM_AGENT={agent_warm.elapsed_seconds:.3f}s")
        print(f"CONTROL_WARM_AGENT_TOKENS={agent_warm.token_count}")
        print(
            f"CONTROL_ENGINE_INIT_MANAGER={bundle.manager_engine_init_seconds:.3f}s"
        )
        print(f"CONTROL_ENGINE_INIT_AGENT={bundle.agent_engine_init_seconds:.3f}s")
        print(
            "CONTROL_ENGINE_INIT="
            f"{bundle.manager_engine_init_seconds + bundle.agent_engine_init_seconds:.3f}s"
        )
        print(f"CONTROL_CHAIN_TOTAL={chain_total:.3f}s")
        return 0 if turn.kind in {"reply", "ask"} else 2
    finally:
        bundle.close()


def _build_bundle(settings: Settings) -> ProbeBundle:
    model_path = Path(os.getenv("LITERT_AGENT_MODEL_PATH", str(DEFAULT_E4B_MODEL)))
    if not model_path.is_file():
        raise FileNotFoundError(f"LiteRT-LM model not found: {model_path}")

    backend_name = os.getenv("LITERT_AGENT_BACKEND", "cpu").strip().lower()
    cpu_threads = _env_optional_positive_int("LITERT_AGENT_CPU_THREADS")
    max_num_tokens = _env_optional_positive_int("LITERT_AGENT_MAX_NUM_TOKENS")
    speculative = _env_bool("LITERT_AGENT_SPECULATIVE", False)

    LOGGER.info("LiteRT 0.15 Session comparison probe starting")
    LOGGER.info("Runtime package version: %s", _runtime_version())
    LOGGER.info("Model path: %s", model_path)
    LOGGER.info("Backend: %s", backend_name)
    LOGGER.info("CPU threads override: %s", cpu_threads or "default")
    LOGGER.info("Max KV tokens override: %s", max_num_tokens or "model default")
    LOGGER.info("Speculative decoding: %s", speculative)
    LOGGER.info("Role KV isolation: manager Engine != agent Engine")
    LOGGER.info("KV strategy: low-level Session native prefill/decode")

    manager_engine, manager_init = _create_engine(
        model_path,
        backend_name,
        cpu_threads,
        max_num_tokens,
        speculative,
        label="manager",
    )
    try:
        agent_engine, agent_init = _create_engine(
            model_path,
            backend_name,
            cpu_threads,
            max_num_tokens,
            speculative,
            label="agent",
        )
    except Exception:
        manager_engine.close()
        raise

    prompt_store = PromptStore(settings.prompt_dir, settings.agent_count)
    prompt_store.validate()
    skill_base = SkillBase(settings.prompt_dir / "prompt_base.txt")

    manager_client = LiteRT015ChatClient(
        manager_engine,
        max_output_tokens=settings.manager_max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
        label="manager",
    )
    agent_client = LiteRT015ChatClient(
        agent_engine,
        max_output_tokens=settings.agent_max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
        label="agent",
    )

    workers = [
        AgentWorker(
            f"agent{index}",
            agent_client,
            prompt_store,
            settings.workspace,
            max_steps=settings.max_agent_steps,
            max_file_bytes=settings.max_file_bytes,
            command_timeout_seconds=settings.command_timeout_seconds,
        )
        for index in range(1, settings.agent_count + 1)
    ]
    runtime = ManagerRuntime(
        manager_client,
        skill_base,
        prompt_store,
        AgentPool(workers),
        max_steps=settings.max_manager_steps,
    )

    return ProbeBundle(
        runtime=runtime,
        manager_engine=manager_engine,
        agent_engine=agent_engine,
        manager_client=manager_client,
        agent_client=agent_client,
        manager_engine_init_seconds=manager_init,
        agent_engine_init_seconds=agent_init,
    )


def _create_engine(
    model_path: Path,
    backend_name: str,
    cpu_threads: int | None,
    max_num_tokens: int | None,
    speculative: bool,
    *,
    label: str,
) -> tuple[litert_lm.Engine, float]:
    kwargs: dict[str, object] = {
        "backend": _backend(backend_name, cpu_threads),
        "max_num_tokens": max_num_tokens,
        "enable_speculative_decoding": speculative,
        "enable_benchmark": True,
    }
    started = time.monotonic()
    engine = litert_lm.Engine(str(model_path), **kwargs)
    engine.__enter__()
    elapsed = time.monotonic() - started
    LOGGER.info("LiteRT 0.15 %s Engine ready in %.3f s", label, elapsed)
    return engine, elapsed


def _backend(name: str, cpu_threads: int | None) -> litert_lm.Backend:
    if name == "cpu":
        return litert_lm.Backend.CPU(thread_count=cpu_threads)
    if name == "gpu":
        return litert_lm.Backend.GPU()
    raise ValueError(
        f"LITERT_AGENT_BACKEND must be 'cpu' or 'gpu', got {name!r}"
    )


def _runtime_version() -> str:
    for package in ("litert-lm", "litert-lm-api"):
        try:
            return f"{package}={metadata.version(package)}"
        except metadata.PackageNotFoundError:
            continue
    return "unknown"


def _env_optional_positive_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


if __name__ == "__main__":
    sys.exit(main())
