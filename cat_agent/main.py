from __future__ import annotations

import logging
import sys

from .agent import AgentWorker
from .config import Settings
from .manager import ManagerRuntime
from .model_client import OpenAIChatClient
from .pool import AgentPool
from .prompt_store import PromptStore
from .skills import SkillBase

LOGGER = logging.getLogger(__name__)


def _client(settings: Settings, *, max_output_tokens: int) -> OpenAIChatClient:
    return OpenAIChatClient(
        api_base_url=settings.api_base_url,
        model=settings.model,
        timeout_seconds=settings.http_timeout_seconds,
        retries=settings.request_retries,
        retry_delay_seconds=settings.retry_delay_seconds,
        max_output_tokens=max_output_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        reasoning_effort=settings.reasoning_effort,
    )


def build_runtime(settings: Settings) -> ManagerRuntime:
    prompt_store = PromptStore(settings.prompt_dir, settings.agent_count)
    prompt_store.validate()
    skill_base = SkillBase(settings.prompt_dir / "prompt_base.txt")

    manager_client = _client(settings, max_output_tokens=settings.manager_max_output_tokens)
    agent_client = _client(settings, max_output_tokens=settings.agent_max_output_tokens)

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
    return ManagerRuntime(
        manager_client,
        skill_base,
        prompt_store,
        AgentPool(workers),
        max_steps=settings.max_manager_steps,
    )


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    LOGGER.info("cat-agent starting")
    LOGGER.info("Workspace: %s", settings.workspace)
    LOGGER.info("Prompt dir: %s", settings.prompt_dir)
    LOGGER.info("Model endpoint: %s model=%s", settings.api_base_url, settings.model)
    LOGGER.info("Agent containers: %d", settings.agent_count)
    LOGGER.info("Model reasoning effort request: %s", settings.reasoning_effort)

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    print("cat-agent manager ready. Commands: /quit")
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

        turn = runtime.user_message(text)
        prefix = "manager" if turn.kind in {"reply", "ask"} else turn.kind
        print(f"{prefix}> {turn.text}")

    LOGGER.info("cat-agent stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
