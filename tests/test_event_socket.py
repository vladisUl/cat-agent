from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest

from litert_agent.event_socket import ExternalEventSocket


class ExternalEventSocketTest(unittest.TestCase):
    def test_datagram_reaches_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.sock"
            received: list[tuple[str, str]] = []
            ready = threading.Event()

            def callback(source: str, name: str) -> None:
                received.append((source, name))
                ready.set()

            server = ExternalEventSocket(callback, path)
            server.start()
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
                try:
                    client.sendto(
                        json.dumps(
                            {"source": "gpio", "name": "task_gpio1"}
                        ).encode("utf-8"),
                        str(path),
                    )
                finally:
                    client.close()

                self.assertTrue(ready.wait(1.0))
                self.assertEqual(received, [("gpio", "task_gpio1")])
            finally:
                server.close()


if __name__ == "__main__":
    unittest.main()
