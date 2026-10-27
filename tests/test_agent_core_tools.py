"""Stage-1 CORE contract tests: require no LiteRT/model/MQTT/Task SYSTEM."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from agent_core_tools.server import CoreServer, DEFAULT_SOCKET
from orchestration_tools.contracts import ManagerTurn


class ToolsCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.server = CoreServer(self.root / "tools.sock")
        self.server.start()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._close_server)

    def _close_server(self):
        self.server.close()
        self.thread.join(timeout=1.5)
        self.assertFalse(self.thread.is_alive())

    def _connect(self):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(2)
        connection.connect(str(self.server.path))
        self.addCleanup(connection.close)
        reader = connection.makefile("rb")
        self.addCleanup(reader.close)
        return connection, reader

    @staticmethod
    def _query(conn, reader, payload):
        conn.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        response = reader.readline()
        if not response:
            raise RuntimeError("server disconnected")
        return json.loads(response)

    def test_core_is_isolated_and_explicitly_not_ready(self):
        conn, reader = self._connect()
        self.assertEqual(self._query(conn, reader, {"type": "ping"}), {"type": "pong"})
        status = self._query(conn, reader, {"type": "snapshot"})
        self.assertEqual(status["type"], "snapshot")
        self.assertEqual(status["status"]["state"], "NOT_READY")
        self.assertFalse(status["status"]["ready"])
        self.assertEqual(self._query(conn, reader, {"type": "who"}), {"type": "session", "owner": None})
        self.assertTrue(stat.S_ISSOCK(self.server.path.stat().st_mode))
        self.assertEqual(stat.S_IMODE(self.server.path.stat().st_mode), 0o660)
        self.assertEqual(DEFAULT_SOCKET, Path("/run/cat-agent/litert-tools.sock"))
        self.assertEqual(ManagerTurn("reply", "ok").text, "ok")

    def test_acquire_busy_user_denied_without_fake_completion_and_release(self):
        one, r1 = self._connect()
        two, r2 = self._connect()
        first = self._query(one, r1, {"type": "acquire", "client": "tui"})
        self.assertEqual(first["type"], "acquired")
        self.assertEqual(first["core"]["model"], None)
        self.assertEqual(first["core"]["backend"], "UNCONNECTED")
        self.assertTrue(first["session_id"])
        busy = self._query(two, r2, {"type": "acquire", "client": "web"})
        self.assertEqual(busy["type"], "busy")
        self.assertEqual(busy["owner"], "tui")
        self.assertEqual(self._query(two, r2, {"type": "user", "text": "hi"})["type"], "busy")
        denied = self._query(one, r1, {"type": "user", "text": "выполни shell ls"})
        self.assertEqual(denied["type"], "error")
        self.assertIn("manager backend", denied["error"])
        self.assertEqual(self._query(one, r1, {"type": "release"})["type"], "released")
        self.assertEqual(self._query(two, r2, {"type": "acquire", "client": "web"})["type"], "acquired")

    def test_disconnect_releases_session_automatically(self):
        one, reader = self._connect()
        self.assertEqual(self._query(one, reader, {"type": "acquire", "client": "tui"})["type"], "acquired")
        one.shutdown(socket.SHUT_RDWR)
        one.close()
        two, r2 = self._connect()
        for _ in range(30):
            if self._query(two, r2, {"type": "who"})["owner"] is None:
                break
            time.sleep(0.01)
        self.assertEqual(self._query(two, r2, {"type": "acquire", "client": "web"})["type"], "acquired")

    def test_batched_partial_and_invalid_frames(self):
        conn, reader = self._connect()
        conn.sendall(b'{"type":"ping"}\n{"type":"who"}\n')
        self.assertEqual(json.loads(reader.readline())["type"], "pong")
        self.assertEqual(json.loads(reader.readline())["type"], "session")
        conn.sendall(b'{"type":"pi')
        conn.sendall(b'ng"}\n')
        self.assertEqual(json.loads(reader.readline())["type"], "pong")
        conn.sendall(b'not_json\n')
        self.assertEqual(json.loads(reader.readline())["type"], "error")
        conn.sendall(b'{"type":"user","text":"\\udcd1"}\n')
        self.assertEqual(json.loads(reader.readline())["type"], "error")
        self.assertEqual(self._query(conn, reader, {"type": "ping"})["type"], "pong")

    def test_unsupported_actions_are_not_executed(self):
        conn, reader = self._connect()
        for typ in ("voice", "event", "register_fallback", "warm_kv2", "unknown"):
            with self.subTest(typ=typ):
                reply = self._query(conn, reader, {"type": typ, "text": "do work"})
                self.assertEqual(reply["type"], "error")
                self.assertIn("stage 1", reply["error"])

    def test_oversized_frame_is_rejected(self):
        conn, reader = self._connect()
        conn.sendall(b"x" * 65537 + b"\n")
        reply = json.loads(reader.readline())
        self.assertEqual(reply["type"], "error")
        self.assertIn("64 KiB", reply["error"])

    def test_cannot_start_two_cores_on_same_socket(self):
        other = CoreServer(self.server.path)
        with self.assertRaises(OSError):
            other.start()
        other.close()
        conn, reader = self._connect()
        self.assertEqual(self._query(conn, reader, {"type": "ping"})["type"], "pong")

    def test_socket_owner_does_not_delete_unrelated_files_or_replaced_path(self):
        path = self.server.path
        path.rename(self.root / "old.sock")
        path.write_text("untouched")
        self.server.close()
        self.assertEqual(path.read_text(), "untouched")
        path.unlink()
        with patch("agent_core_tools.server.DEFAULT_SOCKET", path):
            self.server = CoreServer(path)
            self.server.start()
        self.assertTrue(path.exists())

    def test_stale_socket_recovered_but_ordinary_file_preserved(self):
        path = self.root / "stale.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as old:
            old.bind(str(path))
        fresh = CoreServer(path)
        try:
            fresh.start()
            self.assertTrue(path.exists())
        finally:
            fresh.close()
        path.write_text("ordinary")
        with self.assertRaises(FileExistsError):
            fresh.start()
        self.assertEqual(path.read_text(), "ordinary")


class CliTests(unittest.TestCase):
    def test_cli_uses_separate_path_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            endpoint = Path(root) / "cli.sock"
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
            process = subprocess.Popen(
                [sys.executable, "-m", "agent_core_tools", "--socket", str(endpoint)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            )
            try:
                for _ in range(100):
                    if endpoint.exists():
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(.02)
                self.assertTrue(endpoint.exists())
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    conn.settimeout(2)
                    conn.connect(str(endpoint))
                    conn.sendall(b'{"type":"ping"}\n')
                    self.assertEqual(json.loads(conn.recv(512))["type"], "pong")
            finally:
                process.terminate()
                process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0)
            self.assertFalse(endpoint.exists())


if __name__ == "__main__":
    unittest.main()
