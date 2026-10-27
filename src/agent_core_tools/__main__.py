"""Standalone native-tools CORE: --model/--system-file activate Gemma manager."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import sys

from .server import CoreServer, DEFAULT_SOCKET


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Isolated native-tools CORE')
    parser.add_argument('--socket', type=Path, default=DEFAULT_SOCKET)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--system-file', type=Path)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--max-output-tokens', type=int, default=512)
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--enable-shell', action='store_true')
    parser.add_argument('--shell-cwd', type=Path, default=Path('/opt/model'))
    parser.add_argument('--shell-timeout', type=float, default=15.0)
    parser.add_argument('--shell-max-output', type=int, default=16384)
    parser.add_argument('--disable-mqtt', action='store_true')
    parser.add_argument('--agent-config', type=Path,
                        help='JSON agent pool configuration (experimental)')
    parser.add_argument('--trace-native-json', action='store_true')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    args = parser.parse_args(argv)
    if args.agent_config is not None and args.model is None:
        parser.error('--agent-config requires --model and --system-file')
    if bool(args.model) != bool(args.system_file):
        parser.error('--model and --system-file must be used together')
    logging.basicConfig(level=getattr(logging, args.log_level))
    manager = None
    if args.model:
        from orchestration_tools.litert_runtime import build_native_manager
        try:
            manager = build_native_manager(
                args.model, args.system_file,
                enable_shell=args.enable_shell, shell_cwd=args.shell_cwd,
                shell_timeout=args.shell_timeout, shell_max_output=args.shell_max_output,
                enable_mqtt=not args.disable_mqtt,
                max_context_tokens=args.max_context_tokens,
                max_output_tokens=args.max_output_tokens,
                cpu_threads=args.cpu_threads, temperature=args.temperature,
                top_p=args.top_p, trace_native_json=args.trace_native_json,
                agent_config=args.agent_config,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f'TOOLS CORE manager initialization failed: {exc}', file=sys.stderr)
            return 2
    server = CoreServer(args.socket, manager=manager)

    def terminate(_signal, _frame) -> None:
        server.close()

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.start()
        server.serve_forever()
    except OSError as exc:
        print(f'TOOLS CORE startup/service error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
