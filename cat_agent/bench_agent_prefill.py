from __future__ import annotations

import logging
import os
from statistics import median
import sys
import time

from .command_runtime import CommandResult, CommandRuntime
from .config import Settings
from .main import build_runtime
from .prompt_store import AGENT_BOOTSTRAP_ACK


WARMUP_SKILLS = ("mqtt",)
BENCH_TASK = "Получить текущую температуру на улице."
DEFAULT_BENCH_STDOUT = "22.00\n"


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

    pause_seconds = float(os.getenv("CAT_AGENT_BENCH_PAUSE_SECONDS", "10"))
    if pause_seconds < 0:
        raise ValueError("CAT_AGENT_BENCH_PAUSE_SECONDS must be >= 0")

    bench_stdout = os.getenv("CAT_AGENT_BENCH_STDOUT", DEFAULT_BENCH_STDOUT)
    if not bench_stdout.endswith("\n"):
        bench_stdout += "\n"

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

    prompt_samples: list[float] = []
    second_generate_samples: list[float] = []
    second_total_samples: list[float] = []
    new_tokens: list[int] = []

    print(
        f"agent second-pass prefill benchmark: runs={runs}, pause={pause_seconds:g}s, "
        f"task={BENCH_TASK!r}, stdout={bench_stdout.strip()!r}"
    )

    try:
        for index in range(1, runs + 1):
            # Give the SoC a repeatable idle interval before each sample so
            # consecutive runs do not benchmark accumulated sustained load.
            if pause_seconds:
                time.sleep(pause_seconds)

            # Return slot 1 to the same stable prefix used by warmup_agent.
            client.prefill(warm_messages)

            # The first pass is real model generation so the slot contains the
            # exact generated assistant token sequence, just like production.
            first = client.chat(task_messages)

            result = CommandResult(
                command=first.content.strip(),
                exit_code=0,
                stdout=bench_stdout,
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

            # Use the real production path for the second pass as well. The
            # native response reports prompt and generation timing separately.
            second = client.chat(second_messages)

            if second.prompt_seconds is not None:
                prompt_samples.append(second.prompt_seconds)
            if second.generation_seconds is not None:
                second_generate_samples.append(second.generation_seconds)
            second_total_samples.append(second.elapsed_seconds)
            if second.prompt_evaluated_tokens is not None:
                new_tokens.append(second.prompt_evaluated_tokens)

            print(
                f"run {index}: "
                f"first_new={first.prompt_evaluated_tokens if first.prompt_evaluated_tokens is not None else '?'}, "
                f"first_prefill={_fmt_seconds(first.prompt_seconds)}, "
                f"first_generated={first.completion_tokens if first.completion_tokens is not None else '?'}, "
                f"first_generate={_fmt_seconds(first.generation_seconds)}, "
                f"first_total={first.elapsed_seconds:.3f}s, "
                f"second_new={second.prompt_evaluated_tokens if second.prompt_evaluated_tokens is not None else '?'}, "
                f"second_prefill={_fmt_seconds(second.prompt_seconds)}, "
                f"second_generated={second.completion_tokens if second.completion_tokens is not None else '?'}, "
                f"second_generate={_fmt_seconds(second.generation_seconds)}, "
                f"second_total={second.elapsed_seconds:.3f}s, "
                f"command={first.content.strip()!r}"
            )
    finally:
        # Leave slot 1 in its normal warmed state after the benchmark.
        try:
            client.prefill(warm_messages)
        except Exception:
            pass

    if not prompt_samples:
        print("No native prompt timing samples were returned.", file=sys.stderr)
        return 2

    token_text = (
        str(new_tokens[0])
        if new_tokens and all(value == new_tokens[0] for value in new_tokens)
        else repr(new_tokens)
    )
    generate_text = (
        f", generate_median={median(second_generate_samples):.3f}s"
        if second_generate_samples
        else ""
    )
    print(
        "second-pass summary: "
        f"new={token_text}, "
        f"prefill_min={min(prompt_samples):.3f}s, "
        f"prefill_median={median(prompt_samples):.3f}s, "
        f"prefill_max={max(prompt_samples):.3f}s"
        f"{generate_text}, "
        f"total_median={median(second_total_samples):.3f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
