from __future__ import annotations

import os
import socket


def notify_systemd(*, status: str | None = None, ready: bool = False, stopping: bool = False) -> bool:
    """Send a notification to systemd when NOTIFY_SOCKET is available.

    Outside a Type=notify service this is a no-op. Notification failures must
    never affect the agent itself.
    """
    notify_socket = os.getenv("NOTIFY_SOCKET")
    if not notify_socket:
        return False

    address: str | bytes = notify_socket
    if notify_socket.startswith("@"):
        address = b"\0" + notify_socket[1:].encode()

    fields: list[str] = []
    if status is not None:
        fields.append(f"STATUS={status}")
    if ready:
        fields.append("READY=1")
    if stopping:
        fields.append("STOPPING=1")
    if not fields:
        return False

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        sock.connect(address)
        sock.sendall("\n".join(fields).encode())
        return True
    except OSError:
        return False
    finally:
        sock.close()
