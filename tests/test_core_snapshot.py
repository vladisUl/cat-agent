from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest

from litert_agent.core_server import CoreServer
from litert_agent.model_client import InferenceTiming


@dataclass
class FakeResponse:
    cached_tokens: int = 100
    prompt_evaluated_tokens: int = 20
    completion_tokens: int = 10
    prompt_seconds: float = 0.2
    generation_seconds: float = 0.4
    elapsed_seconds: float = 0.6


class FakeClient:
    def __init__(self, resident: int, finished_at: float) -> None:
        self.resident_tokens = resident
        self.last_response = FakeResponse()
        self.inference_timing = InferenceTiming(
            "idle",
            None,
            0.2,
            0.4,
            0.6,
            finished_at,
        )


@dataclass
class FakeTask:
    task_id: int
    description: str
    method: str


@dataclass
class FakeTimer:
    task_id: int
    period_seconds: float
    enabled: bool
    next_fire_monotonic: float | None


class FakeSystemRuntime:
    def task_snapshot(self):
        return (
            FakeTask(1, "temperature", "query"),
            FakeTask(2, "door", "query"),
        )

    def task_timer_snapshot(self):
        return (
            FakeTimer(1, 60.0, True, time.monotonic() + 30.0),
        )


class FakeRuntime:
    pass


class FakeScheduler:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def status_snapshot(self) -> dict[str, object]:
        return {
            "state": "IDLE",
            "label": "",
            "pending": 0,
            "background": False,
        }


class FakeBundle:
    def __init__(self) -> None:
        now = time.monotonic()
        self.runtime = FakeRuntime()
        self.system_runtime = FakeSystemRuntime()
        self.manager_client = FakeClient(1200, now - 2.0)
        self.agent_clients = (FakeClient(700, now - 1.0),)
        self.model_path = Path("model.litertlm")
        self.backend_name = "cpu"
        self.speculative = False
        self.manager_engine_init_seconds = 1.0
        self.agent_engine_init_seconds = 2.0
        self.manager_warm = None
        self.agent_warm = None

    @property
    def agent_client(self):
        return self.agent_clients[0]


def send(sock: socket.socket, payload: dict[str, object]) -> None:
    sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def recv(reader) -> dict[str, object]:
    item = json.loads(reader.readline())
    assert isinstance(item, dict)
    return item


class CoreSnapshotTest(unittest.TestCase):
    def test_snapshot_contains_model_stats_and_task_timer_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "core.sock"
            server = CoreServer(
                FakeBundle(),
                path=path,
                scheduler=FakeScheduler(),  # type: ignore[arg-type]
            )
            server.start()
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(str(path))
            reader = sock.makefile("r", encoding="utf-8", newline="\n")
            try:
                send(sock, {"type": "acquire", "client": "tui"})
                self.assertEqual(recv(reader)["type"], "acquired")

                send(sock, {"type": "snapshot"})
                reply = recv(reader)
                self.assertEqual(reply["type"], "snapshot")
                status = reply["status"]
                self.assertEqual(status["manager"]["resident_tokens"], 1200)
                self.assertEqual(status["agent"]["resident_tokens"], 700)
                self.assertEqual(status["inference"]["total_seconds"], 0.6)
                self.assertEqual(status["tasks"][0]["timer"]["period_seconds"], 60.0)
                self.assertIsNone(status["tasks"][1]["timer"])
            finally:
                sock.close()
                reader.close()
                server.close()
                thread.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
