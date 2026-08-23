from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from orchestration.manager import ManagerTurn
from orchestration.system_events import SystemEvent
from litert_agent.core_server import CoreServer
from litert_agent.model_client import InferenceTiming


class FakeRuntime:
    def external_event(self, source: str, name: str):
        if name == "missing":
            return None
        return SystemEvent(
            source=source,
            name=name,
            task="",
            created_monotonic=time.monotonic(),
            task_id=1,
        )


class FakeScheduler:
    def __init__(self) -> None:
        self.users: list[str] = []
        self.voice_users: list[str] = []
        self.events: list[tuple[SystemEvent, int, bool]] = []
        self.started = False
        self.active_label = ""

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def submit_user(self, text: str) -> None:
        self.users.append(text)

    def submit_voice(self, text: str) -> None:
        self.voice_users.append(text)

    def active_request_label(self) -> str:
        return self.active_label

    def enqueue_external_event(self, event, *, priority: int, coalesce: bool) -> None:
        self.events.append((event, priority, coalesce))

    def status_snapshot(self) -> dict[str, object]:
        return {"state": "IDLE", "pending": 0, "label": self.active_label}


class FakeBundle:
    def __init__(self) -> None:
        self.runtime = FakeRuntime()
        self.model_path = Path("model.litertlm")
        self.backend_name = "cpu"
        self.speculative = False
        self.manager_engine_init_seconds = 1.0
        self.agent_engine_init_seconds = 2.0
        self.manager_warm = None
        self.agent_warm = None


def send(sock: socket.socket, payload: dict[str, object]) -> None:
    data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def recv(reader) -> dict[str, object]:
    raw = reader.readline()
    if not raw:
        raise RuntimeError("socket closed")
    item = json.loads(raw)
    assert isinstance(item, dict)
    return item


class CoreServerTest(unittest.TestCase):
    def _start(self, root: Path):
        scheduler = FakeScheduler()
        server = CoreServer(
            FakeBundle(),
            path=root / "core.sock",
            scheduler=scheduler,  # type: ignore[arg-type]
        )
        server.start()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, scheduler, thread

    def _connect(self, path: Path):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(path))
        return sock, sock.makefile("r", encoding="utf-8", newline="\n")

    def test_only_one_human_session_can_be_acquired(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server, _scheduler, thread = self._start(root)
            first_sock, first_reader = self._connect(server.path)
            second_sock, second_reader = self._connect(server.path)
            try:
                send(first_sock, {"type": "acquire", "client": "tui"})
                self.assertEqual(recv(first_reader)["type"], "acquired")

                send(second_sock, {"type": "acquire", "client": "telegram"})
                busy = recv(second_reader)
                self.assertEqual(busy["type"], "busy")
                self.assertEqual(busy["owner"], "tui")
            finally:
                first_sock.close()
                second_sock.close()
                first_reader.close()
                second_reader.close()
                server.close()
                thread.join(timeout=1.0)

    def test_user_input_is_accepted_only_from_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server, scheduler, thread = self._start(root)
            sock, reader = self._connect(server.path)
            try:
                send(sock, {"type": "acquire", "client": "tui"})
                recv(reader)
                send(sock, {"type": "user", "text": "привет"})
                for _ in range(50):
                    if scheduler.users:
                        break
                    time.sleep(0.01)
                self.assertEqual(scheduler.users, ["привет"])
            finally:
                sock.close()
                reader.close()
                server.close()
                thread.join(timeout=1.0)

    def test_voice_turn_routes_to_voice_without_replacing_human_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server, scheduler, thread = self._start(root)
            human_sock, human_reader = self._connect(server.path)
            voice_sock, voice_reader = self._connect(server.path)
            try:
                send(human_sock, {"type": "acquire", "client": "tui"})
                self.assertEqual(recv(human_reader)["type"], "acquired")
                self.assertEqual(server.session_owner(), "tui")

                send(
                    voice_sock,
                    {"type": "voice", "client": "voice", "text": "температура"},
                )
                accepted = recv(voice_reader)
                self.assertEqual(accepted["type"], "voice_accepted")
                self.assertEqual(accepted["priority"], -10)
                self.assertEqual(scheduler.voice_users, ["температура"])
                self.assertEqual(server.session_owner(), "tui")

                scheduler.active_label = "voice"
                timing = InferenceTiming(
                    "decode",
                    time.monotonic(),
                    0.5,
                    None,
                    None,
                    None,
                )
                server._deliver_model_event("manager", "chunk", "двадцать", timing)
                model_event = recv(voice_reader)
                self.assertEqual(model_event["type"], "model_event")
                self.assertEqual(model_event["payload"], "двадцать")

                server._deliver_human_turn(ManagerTurn("reply", "двадцать градусов"))
                reply = recv(voice_reader)
                self.assertEqual(reply["type"], "reply")
                self.assertEqual(reply["text"], "двадцать градусов")
                self.assertEqual(server.session_owner(), "tui")

                scheduler.active_label = "user"
                server._deliver_human_turn(ManagerTurn("reply", "обычный ответ"))
                human_reply = recv(human_reader)
                self.assertEqual(human_reply["text"], "обычный ответ")
            finally:
                human_sock.close()
                voice_sock.close()
                human_reader.close()
                voice_reader.close()
                server.close()
                thread.join(timeout=1.0)

    def test_event_enters_same_core_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server, scheduler, thread = self._start(root)
            sock, reader = self._connect(server.path)
            try:
                send(
                    sock,
                    {"type": "event", "source": "gpio", "name": "task_gpio1"},
                )
                reply = recv(reader)
                self.assertEqual(reply["type"], "event_accepted")
                for _ in range(50):
                    if scheduler.events:
                        break
                    time.sleep(0.01)
                self.assertEqual(len(scheduler.events), 1)
                event, priority, coalesce = scheduler.events[0]
                self.assertEqual(event.source, "gpio")
                self.assertEqual(event.name, "task_gpio1")
                self.assertEqual(priority, 10)
                self.assertFalse(coalesce)
            finally:
                sock.close()
                reader.close()
                server.close()
                thread.join(timeout=1.0)

    def test_notification_prefers_human_then_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            server, _scheduler, thread = self._start(root)
            fallback_sock, fallback_reader = self._connect(server.path)
            human_sock, human_reader = self._connect(server.path)
            try:
                send(
                    fallback_sock,
                    {"type": "register_fallback", "client": "telegram"},
                )
                self.assertEqual(recv(fallback_reader)["type"], "fallback_registered")

                server._deliver_notification(ManagerTurn("reply", "one"))
                self.assertEqual(recv(fallback_reader)["text"], "one")

                send(human_sock, {"type": "acquire", "client": "tui"})
                self.assertEqual(recv(human_reader)["type"], "acquired")
                server._deliver_notification(ManagerTurn("reply", "two"))
                self.assertEqual(recv(human_reader)["text"], "two")
            finally:
                fallback_sock.close()
                human_sock.close()
                fallback_reader.close()
                human_reader.close()
                server.close()
                thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
