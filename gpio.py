#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket


DEFAULT_CORE_SOCKET = Path("/run/cat-agent/core.sock")


def send_event(source: str, name: str, socket_path: Path) -> dict[str, object]:
    payload = (
        json.dumps(
            {"type": "event", "source": source, "name": name},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(socket_path))
        sock.sendall(payload)
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        try:
            raw = reader.readline()
        finally:
            reader.close()
        if not raw:
            raise RuntimeError("CORE closed connection without event acknowledgement")
        response = json.loads(raw)
        if not isinstance(response, dict):
            raise RuntimeError("invalid CORE event acknowledgement")
        return response
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send one simulated GPIO interrupt to cat-agent CORE."
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
        default=os.getenv("CAT_AGENT_CORE_SOCKET", str(DEFAULT_CORE_SOCKET)),
        help="cat-agent CORE Unix socket path",
    )
    args = parser.parse_args()

    source = args.source.strip().lower()
    event = args.event.strip()
    socket_path = Path(args.socket_path)
    try:
        response = send_event(source, event, socket_path)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot send event to {socket_path}: {exc}")

    response_type = response.get("type")
    if response_type != "event_accepted":
        parser.error(f"CORE rejected event: {response}")

    print(f"sent {source}:{event} -> {socket_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
