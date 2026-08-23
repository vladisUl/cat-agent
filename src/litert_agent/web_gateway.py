from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
from typing import Any


DEFAULT_CORE_SOCKET = Path("/run/cat-agent/core.sock")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"

WEB_HOST = os.environ.get("CAT_AGENT_WEB_HOST", "0.0.0.0")
HTTP_PORT = int(os.environ.get("CAT_AGENT_HTTP_PORT", "8080"))
WS_PORT = int(os.environ.get("CAT_AGENT_WS_PORT", "8765"))
CORE_SOCKET = Path(os.environ.get("CAT_AGENT_CORE_SOCKET", str(DEFAULT_CORE_SOCKET)))

_ALLOWED_BROWSER_TYPES = {"user", "snapshot", "who", "ping", "acquire", "release"}


def _browser_to_core(raw: str | bytes) -> dict[str, object]:
    if not isinstance(raw, str):
        raise ValueError("binary WebSocket messages are not supported")

    try:
        item = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid JSON") from exc

    if not isinstance(item, dict):
        raise ValueError("message must be a JSON object")

    message_type = str(item.get("type", "")).strip().lower()
    if message_type not in _ALLOWED_BROWSER_TYPES:
        raise ValueError(f"unsupported message type: {message_type or '[empty]'}")

    if message_type == "user":
        text = str(item.get("text", "")).strip()
        if not text:
            raise ValueError("user text must be non-empty")
        return {"type": "user", "text": text}

    if message_type == "acquire":
        return {"type": "acquire", "client": "web"}

    return {"type": message_type}


def _encode(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


class _CoreBridge:
    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.sock: socket.socket | None = None
        self.reader = None
        self._send_web_lock = threading.Lock()
        self._relay_thread: threading.Thread | None = None
        self._stop = threading.Event()

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(CORE_SOCKET))
        self.sock = sock
        self.reader = sock.makefile("r", encoding="utf-8", newline="\n")

        self.send_core({"type": "acquire", "client": "web"})
        raw = self.reader.readline()
        if not raw:
            raise RuntimeError("CORE closed connection during web acquire")
        self._send_web_raw(raw.strip())

        self._stop.clear()
        self._relay_thread = threading.Thread(
            target=self._relay_core_to_web,
            name="cat-agent-web-core-relay",
            daemon=True,
        )
        self._relay_thread.start()

    def send_core(self, payload: dict[str, object]) -> None:
        sock = self.sock
        if sock is None:
            raise RuntimeError("CORE bridge is not connected")
        sock.sendall(_encode(payload))

    def send_web(self, payload: dict[str, object]) -> None:
        self._send_web_raw(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

    def _send_web_raw(self, raw: str) -> None:
        with self._send_web_lock:
            self.websocket.send(raw)

    def _relay_core_to_web(self) -> None:
        reader = self.reader
        if reader is None:
            return
        try:
            for raw in reader:
                if self._stop.is_set():
                    return
                line = raw.strip()
                if line:
                    self._send_web_raw(line)
        except Exception:
            if not self._stop.is_set():
                try:
                    self.send_web(
                        {"type": "disconnected", "text": "CORE connection closed"}
                    )
                except Exception:
                    pass

    def close(self) -> None:
        self._stop.set()
        if self.sock is not None:
            try:
                self.send_core({"type": "release"})
            except Exception:
                pass
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass

        thread = self._relay_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)

        if self.reader is not None:
            try:
                self.reader.close()
            except OSError:
                pass

        self._relay_thread = None
        self.reader = None
        self.sock = None


def _handle_websocket(websocket: Any) -> None:
    bridge = _CoreBridge(websocket)
    try:
        bridge.connect()
        for raw in websocket:
            try:
                payload = _browser_to_core(raw)
            except ValueError as exc:
                bridge.send_web({"type": "error", "error": str(exc)})
                continue
            bridge.send_core(payload)
    except (OSError, RuntimeError) as exc:
        try:
            bridge.send_web({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        bridge.close()


class _QuietStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        del format, args


def _start_http_server() -> ThreadingHTTPServer:
    handler = partial(_QuietStaticHandler, directory=str(WEB_ROOT))
    server = ThreadingHTTPServer((WEB_HOST, HTTP_PORT), handler)
    thread = threading.Thread(
        target=server.serve_forever,
        name="cat-agent-web-http",
        daemon=True,
    )
    thread.start()
    return server


def main() -> None:
    if not (WEB_ROOT / "index.html").is_file():
        raise FileNotFoundError(f"Web UI not found: {WEB_ROOT / 'index.html'}")
    if not CORE_SOCKET.exists():
        raise FileNotFoundError(f"CORE socket not found: {CORE_SOCKET}")

    try:
        from websockets.sync.server import serve
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'websockets' is required; install it into the LiteRT venv"
        ) from exc

    http_server = _start_http_server()
    print(f"WEB_HTTP_READY: http://{WEB_HOST}:{HTTP_PORT}")
    print(f"WEB_WS_READY: ws://{WEB_HOST}:{WS_PORT}")
    print(f"CORE_SOCKET: {CORE_SOCKET}")

    try:
        with serve(_handle_websocket, WEB_HOST, WS_PORT, max_size=64 * 1024) as server:
            server.serve_forever()
    finally:
        http_server.shutdown()
        http_server.server_close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
