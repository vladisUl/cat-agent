from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import logging
import os
from pathlib import Path
import signal
import socket
import socketserver
import struct
import threading

from agent_core.socket_owner import SocketOwner
from orchestration.mqtt_events import MqttEventMonitor
from .client import DEFAULT_TASK_SOCKET
from .engine import TaskEngine, process_identity

LOGGER = logging.getLogger(__name__)


def encode(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(type(value).__name__)


class Handler(socketserver.StreamRequestHandler):
    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def handle(self):
        try:
            raw = self.rfile.readline(1024 * 1024 + 1)
            if not raw.endswith(b'\n') or len(raw) > 1024 * 1024:
                raise ValueError('request exceeds 1 MiB or is incomplete')
            request = json.loads(raw)
            pid, _, _ = struct.unpack('3i', self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            result = self.server.engine.call(request['target'], request['method'],
                request.get('args', []), request.get('kwargs', {}), peer=process_identity(pid))
            response = dict(ok=True, result=result)
        except Exception as exc:
            response = dict(ok=False, error=str(exc), type=type(exc).__name__)
        try:
            self.wfile.write((json.dumps(response, default=encode, ensure_ascii=False, allow_nan=False) + '\n').encode())
        except (BrokenPipeError, ConnectionResetError):
            pass


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    path = Path(os.getenv('CAT_AGENT_TASK_SOCKET', DEFAULT_TASK_SOCKET))
    engine = TaskEngine(Path(os.getenv('CAT_AGENT_TASK_DB', '/var/lib/cat-agent/task-system.sqlite3')))
    owner = SocketOwner(path)
    server = monitor = None
    stop = threading.Event()
    thread = None
    try:
        owner.acquire()
        server = Server(str(path), Handler, bind_and_activate=False)
        server.engine = engine
        server.server_bind()
        owner.bound()
        os.chmod(path, 0o660)
        server.server_activate()
        monitor = MqttEventMonitor(engine.events,
            engine.consume_mqtt,
            active_state_path=Path(__file__).resolve().parents[2] / 'config/mqtt_event_active.json')
        monitor.start()
        thread = threading.Thread(target=server.serve_forever, name='task-system-rpc')
        thread.start()
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())
        LOGGER.info('TASK_SYSTEM_READY: %s', path)
        while not stop.wait(0.25):
            try:
                engine.tick()
            except Exception:
                LOGGER.exception('Task SYSTEM timer polling failed')
    finally:
        if monitor is not None:
            monitor.close()
        if server is not None:
            if thread is not None:
                server.shutdown()
                thread.join()
            server.server_close()
        owner.release()
        engine.close()


if __name__ == '__main__':
    main()
