from __future__ import annotations

from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time
from typing import Any


DEFAULT_CORE_SOCKET = Path("/run/cat-agent/core.sock")
DEFAULT_CREDENTIALS = Path("/etc/cat-agent/firebase/zigbee.json")
DEFAULT_TOKENS = Path("/etc/cat-agent/firebase/tokens.txt")

CORE_SOCKET = Path(os.environ.get("CAT_AGENT_CORE_SOCKET", str(DEFAULT_CORE_SOCKET)))
CREDENTIALS_FILE = Path(
    os.environ.get("CAT_AGENT_FIREBASE_CREDENTIALS", str(DEFAULT_CREDENTIALS))
)
TOKENS_FILE = Path(os.environ.get("CAT_AGENT_FIREBASE_TOKENS", str(DEFAULT_TOKENS)))
NOTIFICATION_TITLE = os.environ.get("CAT_AGENT_FIREBASE_TITLE", "Гена").strip() or "Гена"
CHANNEL_ID = os.environ.get("CAT_AGENT_FIREBASE_CHANNEL_ID", "").strip()
TTL_SECONDS = int(os.environ.get("CAT_AGENT_FIREBASE_TTL", "3600"))
RECONNECT_SECONDS = float(os.environ.get("CAT_AGENT_FIREBASE_RECONNECT", "2"))


def read_tokens(path: Path) -> list[str]:
    """Read JSONL {id, token} records, keeping the last token for each id."""
    tokens_by_id: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"invalid token record in {path}:{line_number}")
            token_id = str(item.get("id", "")).strip()
            token = str(item.get("token", "")).strip()
            if not token_id or not token:
                raise ValueError(
                    f"token record requires non-empty id and token in {path}:{line_number}"
                )
            tokens_by_id[token_id] = token
    return list(tokens_by_id.values())


def _encode(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _decode(raw: str) -> dict[str, Any]:
    item = json.loads(raw)
    if not isinstance(item, dict):
        raise ValueError("CORE message must be a JSON object")
    return item


def _init_firebase():
    try:
        import firebase_admin
        from firebase_admin import credentials, messaging
    except ImportError as exc:
        raise RuntimeError(
            "firebase-admin is required in /opt/cat-agent-firebase-venv"
        ) from exc

    if not CREDENTIALS_FILE.is_file():
        raise FileNotFoundError(f"Firebase credentials not found: {CREDENTIALS_FILE}")

    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(credentials.Certificate(str(CREDENTIALS_FILE)))

    return messaging


def _send_push(messaging, text: str) -> tuple[int, int]:
    tokens = read_tokens(TOKENS_FILE)
    if not tokens:
        raise RuntimeError(f"Firebase token file is empty: {TOKENS_FILE}")

    android_notification = None
    if CHANNEL_ID:
        android_notification = messaging.AndroidNotification(channel_id=CHANNEL_ID)

    android = messaging.AndroidConfig(
        priority="high",
        ttl=timedelta(seconds=TTL_SECONDS),
        notification=android_notification,
    )
    notification = messaging.Notification(
        title=NOTIFICATION_TITLE,
        body=text,
    )

    success = 0
    failure = 0
    for offset in range(0, len(tokens), 500):
        batch = tokens[offset : offset + 500]
        message = messaging.MulticastMessage(
            tokens=batch,
            notification=notification,
            android=android,
        )
        response = messaging.send_each_for_multicast(message)
        success += response.success_count
        failure += response.failure_count
        if response.failure_count:
            for index, result in enumerate(response.responses):
                if not result.success:
                    token = batch[index]
                    print(
                        "FIREBASE_TOKEN_FAILED: "
                        f"token={token[:20]}... error={result.exception}",
                        flush=True,
                    )
    return success, failure


def _register_fallback(sock: socket.socket, reader) -> None:
    sock.sendall(_encode({"type": "register_fallback", "client": "firebase"}))
    raw = reader.readline()
    if not raw:
        raise RuntimeError("CORE closed connection during Firebase registration")
    response = _decode(raw)
    if response.get("type") != "fallback_registered":
        raise RuntimeError(f"CORE rejected Firebase fallback registration: {response}")


def _serve_connection(messaging) -> None:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(str(CORE_SOCKET))
        reader = sock.makefile("r", encoding="utf-8", newline="\n")
        try:
            _register_fallback(sock, reader)
            print(f"FIREBASE_FALLBACK_READY: {CORE_SOCKET}", flush=True)
            for raw in reader:
                line = raw.strip()
                if not line:
                    continue
                item = _decode(line)
                if item.get("type") != "notification":
                    continue
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                try:
                    success, failure = _send_push(messaging, text)
                except Exception as exc:
                    print(f"FIREBASE_SEND_FAILED: {exc}", flush=True)
                    continue
                print(
                    f"FIREBASE_SENT: success={success} failure={failure}",
                    flush=True,
                )
        finally:
            reader.close()
    finally:
        sock.close()


def main() -> None:
    if not TOKENS_FILE.is_file():
        raise FileNotFoundError(f"Firebase token file not found: {TOKENS_FILE}")

    messaging = _init_firebase()
    print(f"FIREBASE_ADMIN_READY: {CREDENTIALS_FILE}", flush=True)
    print(f"FIREBASE_TOKENS: {TOKENS_FILE}", flush=True)
    if CHANNEL_ID:
        print(f"FIREBASE_CHANNEL: {CHANNEL_ID}", flush=True)
    else:
        print("FIREBASE_CHANNEL: app default", flush=True)

    while True:
        try:
            _serve_connection(messaging)
        except KeyboardInterrupt:
            return
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            print(f"FIREBASE_CORE_DISCONNECTED: {exc}", flush=True)
            time.sleep(RECONNECT_SECONDS)


if __name__ == "__main__":
    main()
