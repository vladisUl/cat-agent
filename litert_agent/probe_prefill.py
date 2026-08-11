from __future__ import annotations

import logging
import sys
import time

import litert_lm

from cat_agent.config import Settings
from cat_agent.prompt_store import MANAGER_BOOTSTRAP_ACK

from .native_main import build_native_runtime


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

        # Materialize exactly the manager's stable prefix. The Conversation is
        # created with system + bootstrap user, then the already-known assistant
        # READY message is submitted with max_output_tokens=0. If LiteRT-LM
        # treats zero as prefill-only, token_count must increase while decode
        # remains zero and no generated text is returned.
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
            response = conv.send_message(
                {"role": "assistant", "content": MANAGER_BOOTSTRAP_ACK},
                max_output_tokens=0,
                thinking_config=thinking,
            )
            elapsed = time.monotonic() - started
            after = conv.token_count
            bench = conv.get_benchmark_info()
            text = _text(response)

            print(f"PROBE_BEFORE={before}")
            print(f"PROBE_AFTER={after}")
            print(f"PROBE_PREFILL_TOKENS={bench.last_prefill_token_count}")
            print(f"PROBE_DECODE_TOKENS={bench.last_decode_token_count}")
            print(f"PROBE_RESPONSE={text!r}")
            print(f"PROBE_ELAPSED={elapsed:.3f}s")

            ok = (
                before == 0
                and after > 0
                and bench.last_prefill_token_count > 0
                and bench.last_decode_token_count == 0
                and text == ""
            )
            print(f"PROBE_PREFILL_ONLY={'YES' if ok else 'NO'}")
            return 0 if ok else 2
        finally:
            conv.close()
    finally:
        bundle.close()


if __name__ == "__main__":
    sys.exit(main())
