from __future__ import annotations

import logging
import sys

from .config import Settings
from .main import build_runtime

LOGGER = logging.getLogger(__name__)
MANAGER_SLOT = 0
AGENT_SLOT = 1


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

    print(
        "CACHE PROBE: manager(slot0) -> manager(slot0) -> "
        "agent(slot1) -> manager(slot0)"
    )

    # 1. Manager request #1: cold full manager context in slot 0.
    runtime.messages.append({"role": "user", "content": "привет"})
    response1 = runtime.client.chat(runtime.messages, id_slot=MANAGER_SLOT)
    _log_response("PROBE 1 MANAGER SLOT0", response1)
    runtime.messages.append({"role": "assistant", "content": response1.content})

    # 2. Manager request #2: continue the same manager context in slot 0.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: готов"}
    )
    response2 = runtime.client.chat(runtime.messages, id_slot=MANAGER_SLOT)
    _log_response("PROBE 2 MANAGER SLOT0", response2)
    runtime.messages.append({"role": "assistant", "content": response2.content})

    # 3. Agent request: a separate complete agent context pinned to slot 1.
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
    response3 = worker.client.chat(agent_messages, id_slot=AGENT_SLOT)
    _log_response("PROBE 3 AGENT SLOT1", response3)

    # 4. Return to manager slot 0.  Slot 1 must not have displaced slot 0 KV.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: снова"}
    )
    response4 = runtime.client.chat(runtime.messages, id_slot=MANAGER_SLOT)
    _log_response("PROBE 4 MANAGER SLOT0 AFTER AGENT", response4)

    print("CACHE PROBE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
