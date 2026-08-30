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

from .model_client import OpenAICompatibleChatClient


@dataclass(slots=True)
class OpenAIRuntimeBundle:
    runtime: AssistantManagerRuntime
    system_runtime: SystemRuntime
    manager_client: OpenAICompatibleChatClient
    agent_clients: tuple[OpenAICompatibleChatClient, ...]
    model_path: Path
    backend_name: str = "openai"
    speculative: bool = False
    manager_engine_init_seconds: float = 0.0
    agent_engine_init_seconds: float = 0.0
    manager_warm: None = None
    agent_warm: None = None

    @property
    def agent_client(self) -> OpenAICompatibleChatClient:
        return self.agent_clients[0]

    def close(self) -> None:
        self.manager_client.close()
        for client in self.agent_clients:
            client.close()


def _client(
    settings: Settings,
    *,
    max_output_tokens: int,
    label: str,
) -> OpenAICompatibleChatClient:
    api_key = os.getenv("CAT_AGENT_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    return OpenAICompatibleChatClient(
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
        api_key=api_key,
    )


def build_bundle(settings: Settings) -> OpenAIRuntimeBundle:
    prompt_store = PromptStore(settings.prompt_dir, settings.agent_count)
    prompt_store.validate()
    skill_base = SkillBase(settings.prompt_dir / "prompt_base.txt")
    system_runtime = SystemRuntime(TaskStore(DEFAULT_TASK_FILE))

    manager_client = _client(
        settings,
        max_output_tokens=settings.manager_max_output_tokens,
        label="manager",
    )
    agent_clients = tuple(
        _client(
            settings,
            max_output_tokens=settings.agent_max_output_tokens,
            label=f"agent{index}",
        )
        for index in range(1, settings.agent_count + 1)
    )

    workers = [
        AgentWorker(
            f"agent{index}",
            agent_clients[index - 1],
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

    return OpenAIRuntimeBundle(
        runtime=runtime,
        system_runtime=system_runtime,
        manager_client=manager_client,
        agent_clients=agent_clients,
        model_path=Path(settings.model),
    )
