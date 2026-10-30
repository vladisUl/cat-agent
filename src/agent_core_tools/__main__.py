"""Standalone native-tools CORE: --model/--system-file activate Gemma manager."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import signal
import sys

from .server import CoreServer, DEFAULT_SOCKET
from .native_logging import configure_native_logging


LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Isolated native-tools CORE')
    parser.add_argument('--socket', type=Path, default=DEFAULT_SOCKET)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--system-file', type=Path)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--max-output-tokens', type=int, default=512)
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--vision-backend', choices=('cpu', 'gpu', 'none'), default='cpu')
    parser.add_argument('--activation-dtype', choices=('fp32', 'fp16', 'int16', 'int8'))
    parser.add_argument('--speculative', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--enable-shell', action='store_true')
    parser.add_argument('--shell-cwd', type=Path, default=Path('/opt/model'))
    parser.add_argument('--shell-timeout', type=float, default=15.0)
    parser.add_argument('--shell-max-output', type=int, default=16384)
    parser.add_argument('--disable-mqtt', action='store_true')
    parser.add_argument('--enable-background-probe', action='store_true',
                        help='Allow ephemeral, explicitly sliced background agent jobs')
    parser.add_argument('--enable-hardware-probe', action='store_true',
                        help='Enable IPC event priority diagnostics (not Task SYSTEM)')
    parser.add_argument('--enable-task-worker', action='store_true',
                        help='Connect to authoritative Task SYSTEM RPC (stage 5A)')
    parser.add_argument('--task-socket', type=Path,
                        default=Path(os.getenv('CAT_AGENT_TASK_SOCKET',
                                             '/run/cat-agent/tasks.sock')))
    parser.add_argument('--agent-config', type=Path,
                        help='JSON agent pool configuration (experimental)')
    parser.add_argument('--trace-native-json', action='store_true')
    parser.add_argument(
        '--log-level',
        default=os.getenv('CAT_AGENT_LOG_LEVEL', 'INFO').upper(),
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    )
    args = parser.parse_args(argv)
    if args.enable_background_probe and args.agent_config is None:
        parser.error('--enable-background-probe requires --agent-config')
    if args.enable_hardware_probe and args.agent_config is None:
        parser.error('--enable-hardware-probe requires --agent-config')
    if args.enable_task_worker and args.agent_config is None:
        parser.error('--enable-task-worker requires --agent-config and --model')
    if args.agent_config is not None and args.model is None:
        parser.error('--agent-config requires --model and --system-file')
    if bool(args.model) != bool(args.system_file):
        parser.error('--model and --system-file must be used together')
    if args.model is None:
        # Skeleton-only diagnostics run without the model and file log.
        logging.basicConfig(level=getattr(logging, args.log_level))
    else:
        try:
            log_path = configure_native_logging(args.log_level)
        except (OSError, ValueError) as exc:
            print(f'Cannot initialize LiteRT log: {exc}', file=sys.stderr)
            return 2
        LOGGER.info("cat-agent LiteRT CORE starting")
        LOGGER.info("Log file: %s", log_path)
        LOGGER.info("Model: %s backend=%s vision_backend=%s",
                    args.model, args.backend, args.vision_backend)
        LOGGER.info("Manager prompt: %s agents: %s", args.system_file, args.agent_config)
        LOGGER.info("CORE socket: %s TASK socket: %s", args.socket, args.task_socket)
    if args.enable_task_worker:
        # Check the shared service BEFORE loading and warming the LiteRT
        # Engine. A second local Task SYSTEM must never be started as fallback.
        from task_system.client import TaskRPC
        from orchestration.tasks import TaskStoreError
        try:
            result = TaskRPC(args.task_socket).call('system', 'ping')
            if result != 'TASK_SYSTEM_READY':
                raise TaskStoreError('unexpected TASK SYSTEM handshake')
        except TaskStoreError as exc:
            LOGGER.error("Shared TASK SYSTEM unavailable at %s: %s", args.task_socket, exc)
            print(f'Shared TASK SYSTEM unavailable at {args.task_socket}: {exc}',
                  file=sys.stderr)
            return 2
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
                cpu_threads=args.cpu_threads, backend_name=args.backend,
                vision_backend_name=args.vision_backend,
                activation_dtype=args.activation_dtype,
                speculative=args.speculative, temperature=args.temperature,
                top_p=args.top_p, trace_native_json=args.trace_native_json,
                agent_config=args.agent_config,
                task_socket=args.task_socket if args.enable_task_worker else None,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            LOGGER.exception("TOOLS CORE manager initialization failed")
            print(f'TOOLS CORE manager initialization failed: {exc}', file=sys.stderr)
            return 2
    agent_permissions = None
    if args.enable_task_worker:
        from orchestration_tools.agent_config import effective_agent_tool_names, load_agent_specs
        from litert_agent_tools.mqtt_tools import BUILTIN_MQTT_TOOL_NAMES
        # TASK admission must use the same effective grants installed in
        # each AGENT Conversation: built-in MQTT plus explicit JSON grants.
        shared_names = () if args.disable_mqtt else BUILTIN_MQTT_TOOL_NAMES
        agent_permissions = {
            spec.agent_id: effective_agent_tool_names(spec, shared_names)
            for spec in load_agent_specs(args.agent_config, prompt_dir=args.system_file.parent)
        }
    server = CoreServer(
        args.socket, manager=manager,
        background_enabled=args.enable_background_probe,
        hardware_probe_enabled=args.enable_hardware_probe,
        task_worker_socket=args.task_socket if args.enable_task_worker else None,
        task_agent_permissions=agent_permissions,
    )

    def terminate(_signal, _frame) -> None:
        server.close()

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.start()
        server.serve_forever()
    except (OSError, RuntimeError) as exc:
        LOGGER.exception("TOOLS CORE startup/service error")
        print(f'TOOLS CORE startup/service error: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        LOGGER.info("cat-agent LiteRT CORE stopped")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
