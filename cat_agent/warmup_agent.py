from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any
from urllib import error, request

from .config import Settings
from .main import AGENT_SLOT, build_runtime
from .prompt_store import AGENT_BOOTSTRAP_ACK

LOGGER = logging.getLogger(__name__)
WARMUP_SKILLS = ("mqtt",)


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    http_request = request.Request(
        url,
        data=encoded,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc}") from exc

    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise RuntimeError(f"unexpected response from {url}: {decoded!r}")
    return decoded


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    skills = runtime.skill_base.require(WARMUP_SKILLS)
    bootstrap = runtime.prompt_store.build_agent_bootstrap(skills, settings.workspace)

    # Use the exact same /v1/chat/completions path as the real agent. The final
    # user message is deliberately left open: a real task begins with the same
    # [TASK] prefix and continues from this point. Disable the template's normal
    # assistant-generation suffix because it is mutually exclusive with
    # continue_final_message. With zero completion tokens the server should only
    # evaluate this prefix into slot 1 and generate nothing.
    messages = [
        {
            "role": "system",
            "content": runtime.prompt_store.agent_system_prompt("agent1"),
        },
        {"role": "user", "content": bootstrap},
        {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
        {"role": "user", "content": "[TASK]"},
    ]

    started = time.monotonic()
    response = _post_json(
        f"{settings.api_base_url}/chat/completions",
        {
            "model": settings.model,
            "messages": messages,
            "stream": False,
            "max_completion_tokens": 0,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "reasoning_effort": settings.reasoning_effort,
            "id_slot": AGENT_SLOT,
            "cache_prompt": True,
            "continue_final_message": True,
            "chat_template_kwargs": {"add_generation_prompt": False},
        },
        settings.http_timeout_seconds,
    )
    elapsed = time.monotonic() - started

    timings = response.get("timings")
    cache_n: int | None = None
    prompt_n: int | None = None
    prompt_ms: float | None = None
    if isinstance(timings, dict):
        if isinstance(timings.get("cache_n"), int):
            cache_n = timings["cache_n"]
        if isinstance(timings.get("prompt_n"), int):
            prompt_n = timings["prompt_n"]
        if isinstance(timings.get("prompt_ms"), (int, float)):
            prompt_ms = float(timings["prompt_ms"])

    print(
        "agent warmup: "
        f"{elapsed:.3f}s, "
        f"cached={cache_n if cache_n is not None else '?'}, "
        f"new={prompt_n if prompt_n is not None else '?'}, "
        f"prefill={prompt_ms / 1000.0:.3f}s" if prompt_ms is not None else
        f"agent warmup: {elapsed:.3f}s, cached={cache_n if cache_n is not None else '?'}, "
        f"new={prompt_n if prompt_n is not None else '?'}, prefill=?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
