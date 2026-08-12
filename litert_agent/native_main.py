from __future__ import annotations

from dataclasses import dataclass
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

from .native_model_client import LiteRTNativeChatClient, NativePrefillResult

LOGGER = logging.getLogger(__name__)

DEFAULT_E4B_MODEL = Path(
    "/opt/litert-lm/models/gemma-4-E4B-it/gemma-4-E4B-it.litertlm"
)


@dataclass(slots=True)
class NativeRuntime:
    runtime: ManagerRuntime
    manager_engine: litert_lm.Engine
    agent_engine: litert_lm.Engine
    manager_client: LiteRTNativeChatClient
    agent_client: LiteRTNativeChatClient
    manager_engine_init_seconds: float
    agent_engine_init_seconds: float

    @property
    def engine_init_seconds(self) -> float:
        return self.manager_engine_init_seconds + self.agent_engine_init_seconds

    def close(self) -> None:
        self.manager_client.close()
        self.agent_client.close()
        self.manager_engine.close()
        self.agent_engine.close()


@dataclass(frozen=True, slots=True)
class NativePrefill:
    manager: NativePrefillResult
    agent: NativePrefillResult

    @property
    def total_seconds(self) -> float:
        return self.manager.elapsed_seconds + self.agent.elapsed_seconds


def build_native_runtime(settings: Settings) -> NativeRuntime:
    model_path = Path(
        os.getenv("LITERT_AGENT_MODEL_PATH", str(DEFAULT_E4B_MODEL))
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"LiteRT-LM model not found: {model_path}")

    backend_name = os.getenv("LITERT_AGENT_BACKEND", "cpu").strip().lower()
    cpu_threads = _env_optional_positive_int("LITERT_AGENT_CPU_THREADS")
    max_num_tokens = _env_optional_positive_int("LITERT_AGENT_MAX_NUM_TOKENS")
    speculative = _env_bool("LITERT_AGENT_SPECULATIVE", False)
    benchmark = _env_bool("LITERT_AGENT_BENCHMARK", False)

    LOGGER.info("Initializing two isolated LiteRT-LM 0.14 E4B Engines")
    LOGGER.info("Model path: %s", model_path)
    LOGGER.info("Backend: %s", backend_name)
    LOGGER.info("CPU threads override: %s", cpu_threads or "default")
    LOGGER.info("Max KV tokens override: %s", max_num_tokens or "model default")
    LOGGER.info("Speculative decoding: %s", speculative)
    LOGGER.info("Benchmark mode: %s", benchmark)
    LOGGER.info("Preface prefill: local 0.14 binding enabled")
    LOGGER.info("Role KV isolation: manager Engine != agent Engine")

    manager_engine, manager_engine_init_seconds = _create_engine(
        model_path,
        backend_name,
        cpu_threads,
        max_num_tokens,
        speculative,
        benchmark,
        label="manager",
    )
    try:
        agent_engine, agent_engine_init_seconds = _create_engine(
            model_path,
            backend_name,
            cpu_threads,
            max_num_tokens,
            speculative,
            benchmark,
            label="agent",
        )
    except Exception:
        manager_engine.close()
        raise

    prompt_store = PromptStore(settings.prompt_dir, settings.agent_count)
    prompt_store.validate()
    skill_base = SkillBase(settings.prompt_dir / "prompt_base.txt")

    manager_client = LiteRTNativeChatClient(
        manager_engine,
        max_output_tokens=settings.manager_max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
        label="manager",
    )
    agent_client = LiteRTNativeChatClient(
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
    return NativeRuntime(
        runtime=runtime,
        manager_engine=manager_engine,
        agent_engine=agent_engine,
        manager_client=manager_client,
        agent_client=agent_client,
        manager_engine_init_seconds=manager_engine_init_seconds,
        agent_engine_init_seconds=agent_engine_init_seconds,
    )


def _create_engine(
    model_path: Path,
    backend_name: str,
    cpu_threads: int | None,
    max_num_tokens: int | None,
    speculative: bool,
    benchmark: bool,
    *,
    label: str,
) -> tuple[litert_lm.Engine, float]:
    started = time.monotonic()
    kwargs: dict[str, object] = {
        "backend": _backend(backend_name, cpu_threads),
        "max_num_tokens": max_num_tokens,
        "enable_speculative_decoding": speculative,
    }
    if benchmark:
        kwargs["enable_benchmark"] = True
    engine = litert_lm.Engine(str(model_path), **kwargs)
    engine.__enter__()
    elapsed = time.monotonic() - started
    LOGGER.info("LiteRT-LM %s Engine ready in %.3f s", label, elapsed)
    return engine, elapsed


def prefill_manager_runtime(bundle: NativeRuntime) -> NativePrefillResult:
    manager_messages = bundle.runtime.messages[:3]
    if (
        len(manager_messages) != 3
        or manager_messages[-1].get("role") != "assistant"
        or manager_messages[-1].get("content") != MANAGER_BOOTSTRAP_ACK
    ):
        raise RuntimeError("Unexpected canonical manager bootstrap history")

    result = bundle.manager_client.prefill_prefix(manager_messages)
    LOGGER.info(
        "LiteRT manager prefill ready in %.3f s resident=%d",
        result.elapsed_seconds,
        result.token_count,
    )
    return result


def prefill_native_runtime(
    bundle: NativeRuntime,
    settings: Settings,
    *,
    agent_skills: tuple[str, ...] = ("mqtt",),
) -> NativePrefill:
    """Prefill exact manager and agent stable histories without generation."""

    manager = prefill_manager_runtime(bundle)

    skills = bundle.runtime.skill_base.require(agent_skills)
    bootstrap = bundle.runtime.prompt_store.build_agent_bootstrap(
        skills,
        settings.workspace,
    )
    agent_messages = [
        {
            "role": "system",
            "content": bundle.runtime.prompt_store.agent_system_prompt("agent1"),
        },
        {"role": "user", "content": bootstrap},
        {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
    ]

    agent = bundle.agent_client.prefill_prefix(agent_messages)
    LOGGER.info(
        "LiteRT agent prefill ready in %.3f s resident=%d skills=%s",
        agent.elapsed_seconds,
        agent.token_count,
        ",".join(agent_skills),
    )

    return NativePrefill(manager=manager, agent=agent)


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    LOGGER.info("cat-agent native LiteRT-LM 0.14 runtime starting")
    LOGGER.info("Workspace: %s", settings.workspace)
    LOGGER.info("Prompt dir: %s", settings.prompt_dir)
    LOGGER.info("Native LiteRT-LM skills/presets: disabled")
    LOGGER.info("Role KV: manager=Engine[0], agent=Engine[1]")

    bundle = build_native_runtime(settings)
    try:
        prefill_manager_runtime(bundle)

        print("cat-agent native LiteRT-LM manager ready. Commands: /quit")
        while True:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                continue
            if text in {"/quit", "/exit"}:
                break

            turn = bundle.runtime.user_message(text)
            prefix = "manager" if turn.kind in {"reply", "ask"} else turn.kind
            print(f"{prefix}> {turn.text}")
    finally:
        bundle.close()

    LOGGER.info("cat-agent native LiteRT-LM runtime stopped")
    return 0


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
