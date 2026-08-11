from __future__ import annotations

import logging
import sys
import time

import litert_lm

from cat_agent.config import Settings
from cat_agent.prompt_store import MANAGER_BOOTSTRAP_ACK

from .native_main import build_native_runtime

TASK = "Получить текущую температуру на улице."


def _text(response: object) -> str:
    if not isinstance(response, dict):
        return repr(response)
    content = response.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


def main() -> int:
    settings = Settings.from_env()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    bundle = build_native_runtime(settings)
    try:
        messages = bundle.runtime.messages
        sampler = litert_lm.SamplerConfig(
            top_p=settings.top_p,
            temperature=settings.temperature,
            seed=0,
        )
        thinking = litert_lm.ThinkingConfig(
            enable_thinking=False,
            thinking_token_budget=0,
        )

        # Materialize exactly the manager's stable prefix. LiteRT-LM appears to
        # prefill the supplied assistant READY message and then perform one
        # terminal decode step even when max_output_tokens=0. The decisive test
        # is therefore not decode==0, but whether a subsequent real user turn
        # continues from the resident KV and produces the correct manager action.
        conv = bundle.engine.create_conversation(
            messages=messages[:2],
            automatic_tool_calling=False,
            sampler_config=sampler,
            thinking_config=thinking,
            max_output_tokens=settings.manager_max_output_tokens,
        )
        try:
            before = conv.token_count
            started = time.monotonic()
            warm_response = conv.send_message(
                {"role": "assistant", "content": MANAGER_BOOTSTRAP_ACK},
                max_output_tokens=0,
                thinking_config=thinking,
            )
            warm_elapsed = time.monotonic() - started
            warm_after = conv.token_count
            warm_bench = conv.get_benchmark_info()
            warm_text = _text(warm_response)

            print(f"PROBE_BEFORE={before}")
            print(f"PROBE_AFTER={warm_after}")
            print(f"PROBE_PREFILL_TOKENS={warm_bench.last_prefill_token_count}")
            print(f"PROBE_DECODE_TOKENS={warm_bench.last_decode_token_count}")
            print(f"PROBE_RESPONSE={warm_text!r}")
            print(f"PROBE_ELAPSED={warm_elapsed:.3f}s")

            live_before = conv.token_count
            started = time.monotonic()
            live_response = conv.send_message(
                {"role": "user", "content": TASK},
                max_output_tokens=settings.manager_max_output_tokens,
                thinking_config=thinking,
            )
            live_elapsed = time.monotonic() - started
            live_after = conv.token_count
            live_bench = conv.get_benchmark_info()
            live_text = _text(live_response)

            print(f"PROBE_LIVE_BEFORE={live_before}")
            print(f"PROBE_LIVE_AFTER={live_after}")
            print(f"PROBE_LIVE_PREFILL_TOKENS={live_bench.last_prefill_token_count}")
            print(f"PROBE_LIVE_DECODE_TOKENS={live_bench.last_decode_token_count}")
            print(f"PROBE_LIVE_RESPONSE={live_text!r}")
            print(f"PROBE_LIVE_ELAPSED={live_elapsed:.3f}s")

            ok = (
                before == 0
                and warm_after > 0
                and warm_bench.last_prefill_token_count > 0
                and warm_text == MANAGER_BOOTSTRAP_ACK
                and live_before == warm_after
                and live_bench.last_prefill_token_count < 100
                and live_text.startswith("DELEGATE mqtt")
            )
            print(f"PROBE_WARM_CONTINUATION={'YES' if ok else 'NO'}")
            return 0 if ok else 2
        finally:
            conv.close()
    finally:
        bundle.close()


if __name__ == "__main__":
    sys.exit(main())
