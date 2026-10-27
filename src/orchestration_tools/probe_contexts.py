"""Hardware probe: two independent native Conversations on ONE LiteRT Engine.

Execute in a dedicated process on the target WSL host. No agent CORE or
Task SYSTEM is started. The probe uses no shell, writes no files, and does
not execute model-supplied commands.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import secrets
import sys
import time

from litert_agent_tools.conversation import response_text
from .litert_runtime import build_native_manager


LOGGER = logging.getLogger(__name__)
WORDS = ("КЕДР", "САПФИР", "БАРХАТ", "ЛАНДЫШ", "ЯНТАРЬ", "ГРАНАТ", "МАЛАХИТ", "ОБСИДИАН")


def _words(text: str) -> list[str]:
    return re.findall(r"[а-яё]+", text.casefold())


def run_probe(manager, *, codes: tuple[str, str]) -> tuple[bool, bool]:
    """Perform A->B->A->B; separately report native KV and semantic isolation."""
    chat = manager.chat
    base_kv = chat.base_kv_tokens
    second = chat.fork_dialogue()
    first_kv = chat.working_kv_tokens
    second_kv = second.token_count
    kv_issues: list[str] = []
    semantic_issues: list[str] = []

    def checkpoint(label: str, *, a_expected: int, b_expected: int) -> None:
        actual = (chat.working_kv_tokens, second.token_count, chat.base_resident_kv_tokens)
        expected = (a_expected, b_expected, base_kv)
        print(f"KV {label}: A={actual[0]} B={actual[1]} BASE={actual[2]}", flush=True)
        if actual != expected:
            kv_issues.append(f"{label}: expected {expected}, got {actual}")

    def act(label: str, question: str, send, kv_reader) -> str:
        before = kv_reader()
        started = time.monotonic()
        answer = send(question)
        after = kv_reader()
        print(
            f"{label}: KV {before} -> {after}; {time.monotonic() - started:.2f}s; "
            f"response={answer!r}",
            flush=True,
        )
        if after is None or before is None or after <= before:
            kv_issues.append(f"{label}: KV did not grow: {before} -> {after}")
        return answer

    try:
        print(f"BASE={base_kv}, A clone={first_kv}, B clone={second_kv}", flush=True)
        if base_kv is None or first_kv != base_kv or second_kv != base_kv:
            kv_issues.append("Two clones did not start at the same BASE length")

        code_a, code_b = codes
        intro = (
            "Это проверка памяти отдельного диалога. Твое секретное слово — {word}. "
            "Запомни именно его. Ответь только: ПРИНЯТО."
        )
        question = (
            "Какое секретное слово я сообщил тебе ранее в ЭТОМ диалоге? "
            "Ответь только одним этим словом, без пояснений."
        )

        act("A1", intro.format(word=code_a), chat.ask, lambda: chat.working_kv_tokens)
        a1_kv = chat.working_kv_tokens
        checkpoint("after A1", a_expected=a1_kv, b_expected=base_kv)

        act("B1", intro.format(word=code_b),
            lambda q: response_text(second.send_message(q)), lambda: second.token_count)
        b1_kv = second.token_count
        checkpoint("after B1", a_expected=a1_kv, b_expected=b1_kv)

        a_reply = act("A2", question, chat.ask, lambda: chat.working_kv_tokens)
        a2_kv = chat.working_kv_tokens
        checkpoint("after A2", a_expected=a2_kv, b_expected=b1_kv)

        b_reply = act("B2", question,
            lambda q: response_text(second.send_message(q)), lambda: second.token_count)
        b2_kv = second.token_count
        checkpoint("after B2", a_expected=a2_kv, b_expected=b2_kv)

        for label, expected_code, other_code, answer in (
            ("A", code_a, code_b, a_reply), ("B", code_b, code_a, b_reply),
        ):
            found = _words(answer)
            if found != [expected_code.casefold()]:
                semantic_issues.append(
                    f"{label}: expected only {expected_code!r}, got {answer!r}"
                )
            if other_code.casefold() in found:
                semantic_issues.append(f"{label}: leaked other dialogue's marker")

        print(f"RESULT native KV isolation: {'PASS' if not kv_issues else 'FAIL'}", flush=True)
        for issue in kv_issues:
            print(f"  KV FAILURE: {issue}", flush=True)
        print(f"RESULT model recalls own code: {'PASS' if not semantic_issues else 'INCONCLUSIVE'}", flush=True)
        for issue in semantic_issues:
            print(f"  RECALL: {issue}", flush=True)
        print(
            "NOTE: Token counts alone cannot prove content isolation. "
            "An unexpected model answer can also be a generation error; "
            "inspect both the KV and semantic checks.",
            flush=True,
        )
        return (not kv_issues, not semantic_issues)
    finally:
        second.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe interleaved native KV contexts")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--system-file", required=True, type=Path)
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--disable-mqtt", action="store_true")
    parser.add_argument("--trace-native-json", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    code_a = secrets.choice(WORDS)
    code_b = secrets.choice(tuple(x for x in WORDS if x != code_a))
    print(f"PROBE markers A={code_a}, B={code_b}", flush=True)

    manager = build_native_manager(
        args.model, args.system_file,
        enable_shell=False,
        enable_mqtt=not args.disable_mqtt,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        cpu_threads=args.cpu_threads,
        temperature=0.0,
        trace_native_json=args.trace_native_json,
    )
    try:
        kv_ok, semantic_ok = run_probe(manager, codes=(code_a, code_b))
    finally:
        manager.close()
    return 0 if kv_ok and semantic_ok else (2 if not kv_ok else 3)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.exception("Probe terminated: %s", exc)
        raise SystemExit(1) from exc
