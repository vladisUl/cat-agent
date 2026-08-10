from __future__ import annotations

import logging
import sys

from .config import Settings
from .main import build_runtime

LOGGER = logging.getLogger(__name__)
WARMUP_SKILLS = ("mqtt",)
WARMUP_TASK = "Прогрев контекста. Команды не выполняй. Ответь DONE и словом готов."


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    worker = runtime.pool.get("agent1")
    if worker is None:
        raise RuntimeError("agent1 not found")

    skills = runtime.skill_base.require(WARMUP_SKILLS)
    prompt = runtime.prompt_store.build_agent_prompt(
        "agent1",
        WARMUP_TASK,
        skills,
        settings.workspace,
    )
    messages = [
        {
            "role": "system",
            "content": runtime.prompt_store.agent_system_prompt("agent1"),
        },
        {"role": "user", "content": prompt},
    ]

    response = worker.client.chat(messages)
    print(
        "agent warmup: "
        f"{response.elapsed_seconds:.3f}s, "
        f"prompt_tokens={response.prompt_tokens if response.prompt_tokens is not None else '?'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
