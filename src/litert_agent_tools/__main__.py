"""Standalone native-tools playground: python -m litert_agent_tools."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from .config import MqttConfig
from .conversation import InvalidUserTextError, NativeConversationChat
from .mqtt_tools import create_mqtt_tools
from .shell_tools import DEFAULT_SHELL_CWD, ShellConfig, ShellGate, create_shell_tool


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated LiteRT Conversation/native MQTT tools test")
    parser.add_argument("--model", required=True, type=Path, help="Path to .litertlm model")
    parser.add_argument("--system-file", required=True, type=Path, help="UTF-8 temporary system prompt")
    parser.add_argument("--max-context-tokens", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--cpu-threads", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--warm-message", default="warm", help="One-time BASE warm-up user message")
    parser.add_argument("--trace-native-json", action="store_true",
                        help="Trace all native message roles; dump failed JSON (may contain private data)")
    parser.add_argument("--enable-shell", action="store_true",
                        help="Enable native shell(command) execution as the current OS user")
    parser.add_argument("--shell-cwd", type=Path, default=DEFAULT_SHELL_CWD,
                        help="Shell working directory (default: /opt/model; NOT a sandbox)")
    parser.add_argument("--shell-timeout", type=float, default=15.0,
                        help="Shell command timeout in seconds (max 120)")
    parser.add_argument("--shell-max-output", type=int, default=16384,
                        help="Maximum combined shell output in bytes (max 65536)")
    args = parser.parse_args(argv)
    if args.max_context_tokens < 2 or args.max_output_tokens < 1:
        parser.error("Token limits must be positive")
    if args.cpu_threads is not None and args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if not args.system_file.is_file():
        parser.error(f"System prompt not found: {args.system_file}")
    prompt = args.system_file.read_text(encoding="utf-8")
    if not prompt.strip():
        parser.error("System prompt is empty")

    shell_gate = None
    native_tools = create_mqtt_tools(MqttConfig.from_env())
    if args.enable_shell:
        try:
            shell_config = ShellConfig(
                workdir=args.shell_cwd,
                timeout_seconds=args.shell_timeout,
                max_output_bytes=args.shell_max_output,
            )
        except ValueError as exc:
            parser.error(str(exc))
        shell_gate = ShellGate()
        native_tools.append(create_shell_tool(shell_config, gate=shell_gate))
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        import litert_lm
    except ImportError as exc:
        parser.error(f"LiteRT-LM is not installed in this Python environment: {exc}")

    engine = litert_lm.Engine(
        str(args.model),
        backend=litert_lm.Backend.CPU(thread_count=args.cpu_threads),
        vision_backend=litert_lm.Backend.CPU(),
        max_num_tokens=args.max_context_tokens,
        enable_ynnpack=False,
        enable_benchmark=True,
    )
    engine.__enter__()
    try:
        sampler = litert_lm.SamplerConfig(
            temperature=args.temperature, top_p=args.top_p, seed=0,
        )
        with NativeConversationChat(
            engine, prompt, native_tools,
            sampler_config=sampler,
            max_output_tokens=args.max_output_tokens,
            thinking_config=litert_lm.ThinkingConfig(enable_thinking=False),
            warm_message=args.warm_message,
            trace_native_json=args.trace_native_json,
        ) as chat:
            if shell_gate is not None:
                shell_gate.enabled = True
            print("Native Conversation tools ready. /new resets dialogue, /exit quits.")
            print("Shell tool: ENABLED (full user permissions)" if shell_gate else "Shell tool: disabled")
            print(f"BASE prefilled: {chat.base_prefilled}; BASE KV tokens: {chat.base_kv_tokens}")
            print(f"BASE warm response: {chat.base_warm_reply}")
            while True:
                try:
                    question = input("You> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not question:
                    continue
                if question in ("/exit", "/quit"):
                    break
                if question == "/new":
                    chat.new_dialogue()
                    print("New dialogue from independent BASE clone.")
                    continue
                try:
                    print("Gemma>", chat.ask(question))
                except InvalidUserTextError as exc:
                    # No native call was made, so the working KV is unchanged.
                    print(f"INPUT ERROR: {exc}", file=sys.stderr)
                    continue
                except Exception as exc:
                    print(f"ERROR: {exc}", file=sys.stderr)
                    # A failed native turn may have mutated state. A fresh clone
                    # is safer than trying to continue a partially failed turn.
                    try:
                        chat.new_dialogue()
                    except Exception as reset_exc:
                        print(f"RESET ERROR: {reset_exc}", file=sys.stderr)
                        return 1
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
