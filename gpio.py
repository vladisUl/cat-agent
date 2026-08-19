#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket


DEFAULT_EVENT_SOCKET = Path("/run/cat-agent/events.sock")


def send_event(source: str, name: str, socket_path: Path) -> None:
    payload = json.dumps(
        {"source": source, "name": name},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, str(socket_path))
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one simulated GPIO interrupt to cat-agent."
    )
    parser.add_argument(
        "event",
        nargs="?",
        default="task_gpio1",
        help="event name from events.json (default: task_gpio1)",
    )
    parser.add_argument(
        "--source",
        default="gpio",
        help="event source (default: gpio)",
    )
    parser.add_argument(
        "--socket",
        dest="socket_path",
        default=os.getenv("CAT_AGENT_EVENT_SOCKET", str(DEFAULT_EVENT_SOCKET)),
        help="Unix datagram socket path",
    )
    args = parser.parse_args()

    socket_path = Path(args.socket_path)
    try:
        send_event(args.source.strip().lower(), args.event.strip(), socket_path)
    except OSError as exc:
        parser.error(f"cannot send event to {socket_path}: {exc}")
    print(f"sent {args.source}:{args.event} -> {socket_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
