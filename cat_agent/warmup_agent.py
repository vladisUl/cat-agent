from __future__ import annotations

import logging
import sys

from .config import Settings
from .main import build_runtime
from .prompt_store import AGENT_BOOTSTRAP_ACK

LOGGER = logging.getLogger(__name__)
WARMUP_SKILLS = ("mqtt",)


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

    # Warm the stable agent prefix through the native llama-server client.
    # n_predict=0 evaluates the templated prompt into slot 1 without decoding.
    # The final [TASK] prefix is shared with real agent requests, so the useful
    # stable part of the prompt remains reusable by cache_prompt.
    messages = [
        {
            "role": "system",
            "content": runtime.prompt_store.agent_system_prompt("agent1"),
        },
        {"role": "user", "content": bootstrap},
        {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
        {"role": "user", "content": "[TASK]"},
    ]

    worker = runtime.pool.get("agent1")
    assert worker is not None
    response = worker.client.prefill(messages)

    print(
        "agent warmup: "
        f"{response.elapsed_seconds:.3f}s, "
        f"cached={response.cached_tokens if response.cached_tokens is not None else '?'}, "
        f"new={response.prompt_evaluated_tokens if response.prompt_evaluated_tokens is not None else '?'}, "
        f"prefill={response.prompt_seconds:.3f}s" if response.prompt_seconds is not None else
        f"agent warmup: {response.elapsed_seconds:.3f}s, "
        f"cached={response.cached_tokens if response.cached_tokens is not None else '?'}, "
        f"new={response.prompt_evaluated_tokens if response.prompt_evaluated_tokens is not None else '?'}, prefill=?"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
