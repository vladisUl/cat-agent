from __future__ import annotations

import json
import logging
import sys
import time
import urllib.error
import urllib.request

from .config import Settings
from .main import build_runtime

LOGGER = logging.getLogger(__name__)
SLOT_ID = 0
CACHE_FILENAME = "manager.bin"


def _log_response(label: str, response) -> None:
    LOGGER.info(
        "%s response in %.3f s: prompt_tokens=%s completion_tokens=%s content=%r",
        label,
        response.elapsed_seconds,
        response.prompt_tokens if response.prompt_tokens is not None else "?",
        response.completion_tokens if response.completion_tokens is not None else "?",
        " ".join(response.content.strip().split())[:300],
    )


def _server_root(api_base_url: str) -> str:
    root = api_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


def _slot_action(api_base_url: str, action: str, filename: str) -> dict:
    url = f"{_server_root(api_base_url)}/slots/{SLOT_ID}?action={action}"
    payload = json.dumps({"filename": filename}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"slot {action} failed: HTTP {exc.code}: {body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"slot {action} failed: {exc}") from exc

    elapsed = time.monotonic() - started
    result = json.loads(body)
    LOGGER.info(
        "SLOT %s slot=%d file=%s in %.3f s: %s",
        action.upper(),
        SLOT_ID,
        filename,
        elapsed,
        json.dumps(result, ensure_ascii=False),
    )
    return result


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
        "CACHE PROBE: manager1 -> SAVE -> manager2 -> agent -> "
        "RESTORE -> manager3"
    )

    # 1. Manager request #1: full manager context.
    runtime.messages.append({"role": "user", "content": "привет"})
    response1 = runtime.client.chat(runtime.messages)
    _log_response("PROBE 1 MANAGER", response1)
    runtime.messages.append({"role": "assistant", "content": response1.content})

    # Save the slot exactly after manager request #1.
    _slot_action(settings.api_base_url, "save", CACHE_FILENAME)

    # 2. Manager request #2: normal continuation of the same manager context.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: готов"}
    )
    response2 = runtime.client.chat(runtime.messages)
    _log_response("PROBE 2 MANAGER", response2)
    runtime.messages.append({"role": "assistant", "content": response2.content})

    # 3. Agent request: completely different system prompt + mqtt skill/context.
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

    # Restore the manager KV cache saved after request #1.
    _slot_action(settings.api_base_url, "restore", CACHE_FILENAME)

    # 4. Manager request #3.  The HTTP request still contains the complete
    # manager message history; restored KV should allow llama-server to reuse
    # the prefix represented by manager.bin instead of rebuilding it after the
    # intervening agent request.
    runtime.messages.append(
        {"role": "user", "content": "Скажи только одно слово: снова"}
    )
    response4 = runtime.client.chat(runtime.messages)
    _log_response("PROBE 4 MANAGER AFTER RESTORE", response4)

    print("CACHE PROBE DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
