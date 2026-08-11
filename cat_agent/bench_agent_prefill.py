from __future__ import annotations

import logging
import os
from statistics import median
import sys

from .command_runtime import CommandResult, CommandRuntime
from .config import Settings
from .main import build_runtime
from .prompt_store import AGENT_BOOTSTRAP_ACK


WARMUP_SKILLS = ("mqtt",)
BENCH_TASK = "Получить текущую температуру на улице."
BENCH_STDOUT = "22.00\n"


def _fmt_seconds(value: float | None) -> str:
    return f"{value:.3f}s" if value is not None else "?"


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    runs = int(os.getenv("CAT_AGENT_BENCH_RUNS", "3"))
    if runs < 1:
        raise ValueError("CAT_AGENT_BENCH_RUNS must be >= 1")

    runtime = build_runtime(settings)
    if not runtime.client.wait_until_ready(lambda: False):
        return 1

    skills = runtime.skill_base.require(WARMUP_SKILLS)
    bootstrap = runtime.prompt_store.build_agent_bootstrap(skills, settings.workspace)
    system_prompt = runtime.prompt_store.agent_system_prompt("agent1")
    task_prompt = runtime.prompt_store.build_agent_task(BENCH_TASK)

    warm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": bootstrap},
        {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
        {"role": "user", "content": "[TASK]"},
    ]
    task_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": bootstrap},
        {"role": "assistant", "content": AGENT_BOOTSTRAP_ACK},
        {"role": "user", "content": task_prompt},
    ]

    worker = runtime.pool.get("agent1")
    assert worker is not None
    client = worker.client

    formatter = CommandRuntime(
        settings.workspace,
        WARMUP_SKILLS,
        max_file_bytes=settings.max_file_bytes,
        timeout_seconds=settings.command_timeout_seconds,
    )

    samples: list[float] = []
    new_tokens: list[int] = []

    print(
        f"agent second-pass prefill benchmark: runs={runs}, "
        f"task={BENCH_TASK!r}, synthetic_stdout={BENCH_STDOUT.strip()!r}"
    )

    try:
        for index in range(1, runs + 1):
            # Return slot 1 to the same stable prefix used by warmup_agent.
            client.prefill(warm_messages)

            # The first pass is real model generation so the slot contains the
            # exact generated assistant token sequence, just like production.
            first = client.chat(task_messages)

            result = CommandResult(
                command=first.content.strip(),
                exit_code=0,
                stdout=BENCH_STDOUT,
                stderr="",
                cwd=settings.workspace.resolve(),
                operation="benchmark",
                metadata={},
            )
            second_messages = [
                *task_messages,
                {"role": "assistant", "content": first.content},
                {"role": "user", "content": formatter.format_result(result)},
            ]

            # Use the real production path for the second pass as well.  The
            # native response reports prompt timing separately from decoding,
            # so generation does not contaminate the prefill measurement.
            second = client.chat(second_messages)

            if second.prompt_seconds is not None:
                samples.append(second.prompt_seconds)
            if second.prompt_evaluated_tokens is not None:
                new_tokens.append(second.prompt_evaluated_tokens)

            print(
                f"run {index}: "
                f"first_new={first.prompt_evaluated_tokens if first.prompt_evaluated_tokens is not None else '?'}, "
                f"first_prefill={_fmt_seconds(first.prompt_seconds)}, "
                f"generated={first.completion_tokens if first.completion_tokens is not None else '?'}, "
                f"second_new={second.prompt_evaluated_tokens if second.prompt_evaluated_tokens is not None else '?'}, "
                f"second_prefill={_fmt_seconds(second.prompt_seconds)}, "
                f"second_generated={second.completion_tokens if second.completion_tokens is not None else '?'}, "
                f"command={first.content.strip()!r}"
            )
    finally:
        # Leave slot 1 in its normal warmed state after the benchmark.
        try:
            client.prefill(warm_messages)
        except Exception:
            pass

    if not samples:
        print("No native prompt timing samples were returned.", file=sys.stderr)
        return 2

    token_text = (
        str(new_tokens[0])
        if new_tokens and all(value == new_tokens[0] for value in new_tokens)
        else repr(new_tokens)
    )
    print(
        "second-pass summary: "
        f"new={token_text}, "
        f"min={min(samples):.3f}s, "
        f"median={median(samples):.3f}s, "
        f"max={max(samples):.3f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
