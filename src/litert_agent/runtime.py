from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time

import litert_lm

from cat_agent.agent import AgentWorker
from cat_agent.config import Settings
from cat_agent.manager import ManagerRuntime
from cat_agent.pool import AgentPool
from cat_agent.prompt_store import AGENT_BOOTSTRAP_ACK, MANAGER_BOOTSTRAP_ACK, PromptStore
from cat_agent.skills import SkillBase
from cat_agent.system_events import SystemRuntime

from .model_client import LiteRTChatClient, WarmResult

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = Path(
    "/storage/models/litertlm/gemma-4-E4B-it.litertlm"
)


@dataclass(slots=True)
class LiteRTRuntimeBundle:
    runtime: ManagerRuntime
    system_runtime: SystemRuntime
    manager_engine: litert_lm.Engine
    agent_engine: litert_lm.Engine
    manager_client: LiteRTChatClient
    agent_client: LiteRTChatClient
    manager_engine_init_seconds: float
    agent_engine_init_seconds: float
    model_path: Path
    backend_name: str
    speculative: bool
    manager_warm: WarmResult | None = None
    agent_warm: WarmResult | None = None

    def close(self) -> None:
        self.manager_client.close()
        self.agent_client.close()
        self.manager_engine.close()
        self.agent_engine.close()


def build_bundle(settings: Settings) -> LiteRTRuntimeBundle:
    model_path = Path(os.getenv("LITERT_AGENT_MODEL_PATH", str(DEFAULT_MODEL)))
    if not model_path.is_file():
        raise FileNotFoundError(f"LiteRT-LM model not found: {model_path}")

    backend_name = os.getenv("LITERT_AGENT_BACKEND", "cpu").strip().lower()
    cpu_threads = _env_optional_positive_int("LITERT_AGENT_CPU_THREADS")
    max_num_tokens = _env_optional_positive_int("LITERT_AGENT_MAX_NUM_TOKENS")
    speculative = _env_bool("LITERT_AGENT_SPECULATIVE", False)
    activation_data_type = _env_activation_data_type(
        "LITERT_AGENT_ACTIVATION_DATA_TYPE"
    )
    bench_skills = _env_skill_names("LITERT_AGENT_BENCH_SKILLS")

    LOGGER.info("LiteRT model: %s", model_path)
    LOGGER.info("LiteRT backend: %s", backend_name)
    LOGGER.info("LiteRT speculative decoding: %s", speculative)
    LOGGER.info(
        "LiteRT activation data type: %s",
        activation_data_type.name if activation_data_type is not None else "default",
    )
    LOGGER.info(
        "LiteRT benchmark skills: %s",
        ",".join(bench_skills) if bench_skills is not None else "disabled",
    )

    manager_engine, manager_init = _create_engine(
        model_path,
        backend_name,
        cpu_threads,
        max_num_tokens,
        speculative,
        activation_data_type,
        label="manager",
    )
    try:
        agent_engine, agent_init = _create_engine(
            model_path,
            backend_name,
            cpu_threads,
            max_num_tokens,
            speculative,
            activation_data_type,
            label="agent",
        )
    except Exception:
        manager_engine.close()
        raise

    prompt_store = PromptStore(settings.prompt_dir, settings.agent_count)
    prompt_store.validate()
    skill_base = SkillBase(settings.prompt_dir / "prompt_base.txt")
    if bench_skills is not None:
        skill_base.require(bench_skills)
    system_runtime = SystemRuntime()

    manager_client = LiteRTChatClient(
        manager_engine,
        max_output_tokens=settings.manager_max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
        label="manager",
        allow_prefix_reset=False,
    )
    agent_client = LiteRTChatClient(
        agent_engine,
        max_output_tokens=settings.agent_max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
        label="agent",
        allow_prefix_reset=True,
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
        system_runtime,
        max_steps=settings.max_manager_steps,
        forced_delegate_skills=bench_skills,
    )

    return LiteRTRuntimeBundle(
        runtime=runtime,
        system_runtime=system_runtime,
        manager_engine=manager_engine,
        agent_engine=agent_engine,
        manager_client=manager_client,
        agent_client=agent_client,
        manager_engine_init_seconds=manager_init,
        agent_engine_init_seconds=agent_init,
        model_path=model_path,
        backend_name=backend_name,
        speculative=speculative,
    )


def warm_bundle(
    bundle: LiteRTRuntimeBundle,
    settings: Settings,
    *,
    default_agent_skill: str = "mqtt",
) -> tuple[WarmResult, WarmResult]:
    manager_messages = bundle.runtime.messages[:3]
    if (
        len(manager_messages) != 3
        or manager_messages[-1].get("role") != "assistant"
        or manager_messages[-1].get("content") != MANAGER_BOOTSTRAP_ACK
    ):
        raise RuntimeError("Unexpected canonical manager bootstrap history")
    manager_warm = bundle.manager_client.prepare_prefix(manager_messages)

    skill_names = bundle.runtime.forced_delegate_skills or (default_agent_skill,)
    skills = bundle.runtime.skill_base.require(skill_names)
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
    bundle.manager_warm = manager_warm
    bundle.agent_warm = agent_warm
    return manager_warm, agent_warm


def _create_engine(
    model_path: Path,
    backend_name: str,
    cpu_threads: int | None,
    max_num_tokens: int | None,
    speculative: bool,
    activation_data_type: litert_lm.ActivationDataType | None,
    *,
    label: str,
) -> tuple[litert_lm.Engine, float]:
    kwargs: dict[str, object] = {
        "backend": _backend(backend_name, cpu_threads),
        "max_num_tokens": max_num_tokens,
        "enable_speculative_decoding": speculative,
        "enable_benchmark": True,
    }
    if activation_data_type is not None:
        kwargs["activation_data_type"] = activation_data_type

    started = time.monotonic()
    engine = litert_lm.Engine(str(model_path), **kwargs)
    engine.__enter__()
    elapsed = time.monotonic() - started
    LOGGER.info("LiteRT %s Engine ready in %.3f s", label, elapsed)
    return engine, elapsed


def _backend(name: str, cpu_threads: int | None) -> litert_lm.Backend:
    if name == "cpu":
        return litert_lm.Backend.CPU(thread_count=cpu_threads)
    if name == "gpu":
        return litert_lm.Backend.GPU()
    raise ValueError(
        f"LITERT_AGENT_BACKEND must be 'cpu' or 'gpu', got {name!r}"
    )


def _env_optional_positive_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {value}")
    return value


def _env_activation_data_type(name: str) -> litert_lm.ActivationDataType | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    value = litert_lm.ActivationDataType.from_str(raw)
    if value is None:
        raise ValueError(
            f"{name} must be one of fp32, fp16, int16, int8; got {raw!r}"
        )
    return value


def _env_skill_names(name: str) -> tuple[str, ...] | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        return None
    if len(set(names)) != len(names):
        raise ValueError(f"{name} contains duplicate skill names: {raw!r}")
    return names


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
