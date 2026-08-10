from __future__ import annotations

import logging
import sys

from .config import Settings
from .main import build_runtime

LOGGER = logging.getLogger(__name__)


def _log_response(label: str, response) -> None:
    LOGGER.info(
        "%s response in %.3f s: prompt_tokens=%s completion_tokens=%s content=%r",
        label,
        response.elapsed_seconds,
        response.prompt_tokens if response.prompt_tokens is not None else "?",
        response.completion_tokens if response.completion_tokens is not None else "?",
        " ".join(response.content.strip().split())[:300],
    )


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    print("CACHE PROBE: 1=manager hello, 2=manager ready, 3=agent mqtt context")

    # Request 1: full manager system prompt + bootstrap + first user message.
    runtime.messages.append({"role": "user", "content": "привет"})
    response1 = runtime.client.chat(runtime.messages)
    _log_response("PROBE 1 MANAGER", response1)
    runtime.messages.append({"role": "assistant", "content": response1.content})

    # Request 2: same manager conversation, only a short new user tail.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: готов"}
    )
    response2 = runtime.client.chat(runtime.messages)
    _log_response("PROBE 2 MANAGER", response2)
    runtime.messages.append({"role": "assistant", "content": response2.content})

    # Request 3: a completely separate real agent prompt.  Use the same mqtt
    # skill/context that the temperature task normally gives agent1, but do not
    # execute any command: this probe measures only the context switch/prefill.
    skills = runtime.skill_base.require(("mqtt",))
    agent_prompt = runtime.prompt_store.build_agent_prompt(
        "agent1",
        "Команды выполнять не требуется. Ответь DONE и одним словом сообщи, что готов.",
        skills,
        settings.workspace,
    )
    agent_messages = [
        {
            "role": "system",
            "content": runtime.prompt_store.agent_system_prompt("agent1"),
        },
        {"role": "user", "content": agent_prompt},
    ]
    worker = runtime.pool.get("agent1")
    if worker is None:
        raise RuntimeError("agent1 not found")
    response3 = worker.client.chat(agent_messages)
    _log_response("PROBE 3 AGENT", response3)

    print("CACHE PROBE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
