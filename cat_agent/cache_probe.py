from __future__ import annotations

import logging
import sys

from .config import Settings
from .main import build_runtime

MANAGER_SLOT = 0
AGENT_SLOT = 1


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}s"


def _print_summary(results: list[tuple[str, object]]) -> None:
    print("\nTIMINGS")
    print("#  role/slot                  wall      prefill    generate   prompt  cached")
    for index, (label, response) in enumerate(results, 1):
        prompt_tokens = "-" if response.prompt_tokens is None else str(response.prompt_tokens)
        cached_tokens = "-" if response.cached_tokens is None else str(response.cached_tokens)
        print(
            f"{index:<2} {label:<26} "
            f"{response.elapsed_seconds:>8.3f}s  "
            f"{_fmt(response.prompt_seconds):>9}  "
            f"{_fmt(response.generation_seconds):>9}  "
            f"{prompt_tokens:>6}  {cached_tokens:>6}"
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

    print("CACHE PROBE: M0 -> M0 -> A1 -> A1 -> M0 -> A1")
    results: list[tuple[str, object]] = []

    # 1. Manager, cold slot 0.
    runtime.messages.append({"role": "user", "content": "привет"})
    response = runtime.client.chat(runtime.messages, id_slot=MANAGER_SLOT)
    results.append(("MANAGER slot0", response))
    runtime.messages.append({"role": "assistant", "content": response.content})

    # 2. Manager again, same slot and conversation.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: готов"}
    )
    response = runtime.client.chat(runtime.messages, id_slot=MANAGER_SLOT)
    results.append(("MANAGER slot0", response))
    runtime.messages.append({"role": "assistant", "content": response.content})

    # Build one independent agent conversation for slot 1.
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

    # 3. Agent, cold slot 1.
    response = worker.client.chat(agent_messages, id_slot=AGENT_SLOT)
    results.append(("AGENT slot1", response))
    agent_messages.append({"role": "assistant", "content": response.content})

    # 4. Agent again, same slot and conversation.
    agent_messages.append(
        {"role": "user", "content": "Ответь DONE и одним словом: второй"}
    )
    response = worker.client.chat(agent_messages, id_slot=AGENT_SLOT)
    results.append(("AGENT slot1", response))
    agent_messages.append({"role": "assistant", "content": response.content})

    # 5. Return to manager slot 0.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: снова"}
    )
    response = runtime.client.chat(runtime.messages, id_slot=MANAGER_SLOT)
    results.append(("MANAGER slot0", response))
    runtime.messages.append({"role": "assistant", "content": response.content})

    # 6. Return to agent slot 1.
    agent_messages.append(
        {"role": "user", "content": "Ответь DONE и одним словом: третий"}
    )
    response = worker.client.chat(agent_messages, id_slot=AGENT_SLOT)
    results.append(("AGENT slot1", response))

    _print_summary(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
