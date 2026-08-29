from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from orchestration.agent import AgentWorker
from orchestration.assistant_manager import AssistantManagerRuntime
from orchestration.config import Settings
from orchestration.pool import AgentPool
from orchestration.prompt_store import PromptStore
from orchestration.skills import SkillBase
from orchestration.system_events import SystemRuntime
from orchestration.tasks import DEFAULT_TASK_FILE, TaskStore

from .model_client import LlamaChatClient, WarmResult

MANAGER_SLOT = 0
AGENT_SLOT = 1


@dataclass(slots=True)
class LlamaRuntimeBundle:
    runtime: AssistantManagerRuntime
    system_runtime: SystemRuntime
    manager_client: LlamaChatClient
    agent_clients: tuple[LlamaChatClient, ...]
    model_path: Path
    backend_name: str = "llama.cpp"
    speculative: bool = False
    manager_engine_init_seconds: float = 0.0
    agent_engine_init_seconds: float = 0.0
    manager_warm: WarmResult | None = None
    agent_warm: WarmResult | None = None

    @property
    def agent_client(self) -> LlamaChatClient:
        return self.agent_clients[0]

    def close(self) -> None:
        self.manager_client.close()
        for client in self.agent_clients:
            if client is not self.manager_client:
                client.close()


def _client(
    settings: Settings,
    *,
    max_output_tokens: int,
    label: str,
    id_slot: int,
) -> LlamaChatClient:
    return LlamaChatClient(
        api_base_url=settings.api_base_url,
        model=settings.model,
        timeout_seconds=settings.http_timeout_seconds,
        retries=settings.request_retries,
        retry_delay_seconds=settings.retry_delay_seconds,
        max_output_tokens=max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
        label=label,
        id_slot=id_slot,
    )


def build_bundle(settings: Settings) -> LlamaRuntimeBundle:
    prompt_store = PromptStore(settings.prompt_dir, settings.agent_count)
    prompt_store.validate()
    skill_base = SkillBase(settings.prompt_dir / "prompt_base.txt")
    system_runtime = SystemRuntime(TaskStore(DEFAULT_TASK_FILE))

    manager_client = _client(
        settings,
        max_output_tokens=settings.manager_max_output_tokens,
        label="manager",
        id_slot=MANAGER_SLOT,
    )
    # llama-server runs two fixed slots: manager=0 and agent=1. All agent
    # containers intentionally share slot 1; CORE serializes model TT calls.
    agent_client = _client(
        settings,
        max_output_tokens=settings.agent_max_output_tokens,
        label="agent",
        id_slot=AGENT_SLOT,
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
    event_worker_id = (
        f"agent{settings.agent_count}" if settings.agent_count > 1 else None
    )
    runtime = AssistantManagerRuntime(
        manager_client,
        skill_base,
        prompt_store,
        AgentPool(workers, event_worker_id=event_worker_id),
        system_runtime,
        max_steps=settings.max_manager_steps,
    )

    speculative = os.getenv("CAT_AGENT_LLAMA_SPECULATIVE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return LlamaRuntimeBundle(
        runtime=runtime,
        system_runtime=system_runtime,
        manager_client=manager_client,
        agent_clients=(agent_client,),
        model_path=Path(settings.model),
        speculative=speculative,
    )


def warm_bundle(
    bundle: LlamaRuntimeBundle,
    settings: Settings,
    *,
    default_agent_skill: str = "mqtt",
) -> tuple[WarmResult, WarmResult]:
    manager_messages = bundle.runtime.messages[:]
    if (
        len(manager_messages) != 1
        or manager_messages[0].get("role") != "system"
    ):
        raise RuntimeError("Unexpected canonical manager system base")
    manager_warm = bundle.manager_client.prepare_prefix(manager_messages)

    skills = bundle.runtime.skill_base.require((default_agent_skill,))
    agent_system_context = bundle.runtime.prompt_store.build_agent_system_context(
        "agent1",
        skills,
        settings.workspace,
    )
    agent_messages = [
        {"role": "system", "content": agent_system_context},
    ]
    agent_warm = bundle.agent_client.prepare_prefix(agent_messages)
    bundle.manager_warm = manager_warm
    bundle.agent_warm = agent_warm
    return manager_warm, agent_warm
