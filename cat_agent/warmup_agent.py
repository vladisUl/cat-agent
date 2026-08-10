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
SENTINEL_A = "A_CAT_AGENT_TASK_BOUNDARY_9f3e"
SENTINEL_B = "Я_CAT_AGENT_TASK_BOUNDARY_4c71"


def _server_root(api_base_url: str) -> str:
    root = api_base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/")


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


def _apply_template(
    root: str,
    messages: list[dict[str, str]],
    timeout: int,
) -> str:
    response = _post_json(
        f"{root}/apply-template",
        {"messages": messages},
        timeout,
    )
    prompt = response.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError(f"/apply-template returned no prompt: {response!r}")
    return prompt


def _tokenize(root: str, text: str, timeout: int) -> list[int]:
    response = _post_json(
        f"{root}/tokenize",
        {
            "content": text,
            "add_special": True,
            "parse_special": True,
        },
        timeout,
    )
    tokens = response.get("tokens")
    if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"/tokenize returned invalid tokens: {response!r}")
    return tokens


def _common_prefix(left: list[int], right: list[int]) -> list[int]:
    size = min(len(left), len(right))
    index = 0
    while index < size and left[index] == right[index]:
        index += 1
    if index == 0:
        raise RuntimeError("agent warmup prompts have no common token prefix")
    return left[:index]


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
    bootstrap = runtime.prompt_store.build_agent_bootstrap(skills, settings.workspace)
    base_messages = [
        {
            "role": "system",
            "content": runtime.prompt_store.agent_system_prompt("agent1"),
        },
        {"role": "user", "content": bootstrap},
        {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
    ]

    # Render two otherwise identical conversations whose TASK contents begin
    # differently. Their token-level longest common prefix is therefore the
    # exact immutable chat-template prefix immediately before variable TASK
    # content. No fake task or generated answer is stored after that boundary.
    messages_a = base_messages + [
        {"role": "user", "content": runtime.prompt_store.build_agent_task(SENTINEL_A)}
    ]
    messages_b = base_messages + [
        {"role": "user", "content": runtime.prompt_store.build_agent_task(SENTINEL_B)}
    ]

    root = _server_root(settings.api_base_url)
    timeout = settings.http_timeout_seconds
    prompt_a = _apply_template(root, messages_a, timeout)
    prompt_b = _apply_template(root, messages_b, timeout)
    tokens_a = _tokenize(root, prompt_a, timeout)
    tokens_b = _tokenize(root, prompt_b, timeout)
    prefix_tokens = _common_prefix(tokens_a, tokens_b)

    started = time.monotonic()
    response = _post_json(
        f"{root}/completion",
        {
            "prompt": prefix_tokens,
            "n_predict": 0,
            "id_slot": AGENT_SLOT,
            "cache_prompt": True,
        },
        timeout,
    )
    elapsed = time.monotonic() - started

    tokens_cached = response.get("tokens_cached")
    tokens_evaluated = response.get("tokens_evaluated")
    print(
        "agent warmup: "
        f"{elapsed:.3f}s, "
        f"prefix_tokens={len(prefix_tokens)}, "
        f"cached={tokens_cached if isinstance(tokens_cached, int) else '?'}, "
        f"evaluated={tokens_evaluated if isinstance(tokens_evaluated, int) else '?'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
