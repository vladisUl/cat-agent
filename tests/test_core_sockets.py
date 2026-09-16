from pathlib import Path
import os
import socket
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent_core.core_server import CoreServer
from agent_core.socket_owner import SocketOwner


class CoreSocketsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"CAT_AGENT_CORE_SOCKET": ""})
        self.env.start()
        self.addCleanup(self.env.stop)

    def server(self, name):
        server = CoreServer(SimpleNamespace(), path=self.root / name,
                            scheduler=Mock(), outbox_path=self.root / (name + ".db"))
        server.outbox = Mock()
        self.addCleanup(server.close)
        return server

    def test_two_cores_keep_separate_endpoints(self):
        litert = self.server("litert.sock")
        openai = self.server("openai.sock")
        litert.start()
        openai.start()
        for server in (litert, openai):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.connect(str(server.path))
                accepted, _ = server._server_socket.accept()
                with accepted:
                    accepted.sendall(server.path.name.encode())
                    self.assertEqual(client.recv(100).decode(), server.path.name)
        litert.close()
        self.assertFalse(litert.path.exists())
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(openai.path))
        self.assertTrue(openai.path.exists())

    def test_duplicate_start_and_close_leave_owner_alive(self):
        owner = self.server("litert.sock")
        other = self.server("litert.sock")
        owner.start()
        with self.assertRaises(OSError):
            other.start()
        other.close()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(owner.path))

    def test_never_started_close_preserves_file(self):
        server = self.server("ordinary-file")
        server.path.write_text("keep")
        with self.assertRaises(FileExistsError):
            server.start()
        server.close()
        self.assertEqual(server.path.read_text(), "keep")

    def test_active_socket_without_lock_is_preserved(self):
        server = self.server("legacy.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as owner:
            owner.bind(str(server.path))
            owner.listen(4)
            inode = server.path.stat().st_ino
            with self.assertRaises(OSError):
                server.start()
            server.close()
            self.assertEqual(server.path.stat().st_ino, inode)

    def test_stale_socket_can_be_restarted(self):
        server = self.server("stale.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(server.path))
        server.start()
        server.close()
        server.start()
        self.assertTrue(server.path.exists())

    def test_close_preserves_replacement(self):
        server = self.server("replaced.sock")
        server.start()
        server.path.rename(self.root / "old.sock")
        server.path.write_text("replacement")
        server.close()
        self.assertEqual(server.path.read_text(), "replacement")

    def test_startup_failure_releases_ownership(self):
        server = self.server("failure.sock")
        server.scheduler.start.side_effect = RuntimeError("warmup failed")
        with self.assertRaises(RuntimeError):
            server.start()
        self.assertFalse(server.path.exists())
        other = self.server("failure.sock")
        other.start()

    def test_environment_overrides_backend_path(self):
        with patch.dict(os.environ, {"CAT_AGENT_CORE_SOCKET": str(self.root / "debug.sock")}):
            server = self.server("litert.sock")
        self.assertEqual(server.path, self.root / "debug.sock")

    def test_lock_excludes_other_owner_until_released(self):
        first = SocketOwner(self.root / "litert.sock")
        second = SocketOwner(first.path)
        self.addCleanup(first.release)
        self.addCleanup(second.release)
        first.acquire()
        with self.assertRaises(BlockingIOError):
            second.acquire()
        second.release()
        with self.assertRaises(BlockingIOError):
            second.acquire()
        first.release()
        second.acquire()

    def test_bind_failure_preserves_foreign_path_and_releases_lock(self):
        server = self.server("bind.sock")

        def competing_bind(_):
            server.path.write_text("foreign")
            raise OSError("address occupied")

        fake_socket = Mock()
        fake_socket.bind.side_effect = competing_bind
        with patch("agent_core.core_server.socket.socket", return_value=fake_socket):
            with self.assertRaises(OSError):
                server.start()
        server.close()
        self.assertEqual(server.path.read_text(), "foreign")
        server.path.unlink()
        owner = SocketOwner(server.path)
        self.addCleanup(owner.release)
        owner.acquire()

    def test_owned_inode_cleanup_and_replacement_preservation(self):
        owner = SocketOwner(self.root / "owned.sock")
        self.addCleanup(owner.release)
        owner.acquire()
        owner.path.touch()
        owner.bound()
        owner.release()
        self.assertFalse(owner.path.exists())
        owner.acquire()
        owner.path.touch()
        owner.bound()
        owner.path.rename(self.root / "original")
        owner.path.write_text("replacement")
        owner.release()
        self.assertEqual(owner.path.read_text(), "replacement")


class LauncherSelectionTest(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def select(self, args, override=""):
        env = dict(os.environ, CAT_AGENT_CORE_SOCKET=override)
        return subprocess.run(
            ["bash", "-c", 'source "$1" "${@:2}"; printf "%s" "$CAT_AGENT_CORE_SOCKET"',
             "launcher", str(self.root / "scripts/select_core.sh"), *args],
            env=env, text=True, capture_output=True)

    def test_backend_selection(self):
        for backend in ("litert", "openai"):
            result = self.select([backend])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, f"/run/cat-agent/{backend}.sock")

    def test_override(self):
        result = self.select(["openai"], "/tmp/debug.sock")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "/tmp/debug.sock")

    def test_no_argument_accepts_override(self):
        result = self.select([], "/tmp/debug.sock")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "/tmp/debug.sock")

    def test_invalid_arguments_fail_even_with_override(self):
        for args in (["unknown"], ["litert", "openai"]):
            result = self.select(args, "/tmp/debug.sock")
            self.assertEqual(result.returncode, 2)

    def test_launchers_validate_before_runtime(self):
        for name in ("start_tui.sh", "start_web.sh", "start_voice.sh"):
            result = subprocess.run(["bash", str(self.root / name), "unknown"],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("Unknown CORE", result.stderr)
