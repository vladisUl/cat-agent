from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from agent_core.outbox import NotificationOutbox


class Owner:
    def __init__(self, path: str, sink: list[tuple[str, str]], *, human: bool) -> None:
        self.path = Path(path)
        self._human = object() if human else None
        self.sink = sink

    def send(self, identifier: str, text: str) -> bool:
        self.sink.append((identifier, text))
        return self._human is None


class NotificationOutboxScopeTest(unittest.TestCase):
    def test_notification_from_other_core_goes_to_active_human(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outbox.sqlite3"
            openai_sent: list[tuple[str, str]] = []
            litert_sent: list[tuple[str, str]] = []
            openai_owner = Owner(
                "/run/cat-agent/openai.sock", openai_sent, human=True
            )
            litert_owner = Owner(
                "/run/cat-agent/litert.sock", litert_sent, human=False
            )
            openai = NotificationOutbox(path, openai_owner.send)
            litert = NotificationOutbox(path, litert_owner.send)

            now = time.time()
            openai.pump(now=now)
            litert.pump(now=now)
            identifier = litert.enqueue("done")
            due = now + 1.0

            self.assertFalse(litert.pump(now=due))
            self.assertEqual(litert_sent, [])

            self.assertTrue(openai.pump(now=due))
            self.assertEqual(openai_sent, [(identifier, "done")])
            openai.acknowledge(identifier)
            self.assertFalse(litert.pump(now=due + 1.0))

    def test_without_human_only_one_live_core_performs_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outbox.sqlite3"
            openai_sent: list[tuple[str, str]] = []
            litert_sent: list[tuple[str, str]] = []
            openai_owner = Owner(
                "/run/cat-agent/openai.sock", openai_sent, human=False
            )
            litert_owner = Owner(
                "/run/cat-agent/litert.sock", litert_sent, human=False
            )
            openai = NotificationOutbox(path, openai_owner.send)
            litert = NotificationOutbox(path, litert_owner.send)

            now = time.time()
            openai.pump(now=now)
            litert.pump(now=now)
            identifier = openai.enqueue("push")
            due = now + 1.0

            # Deterministic election by scope chooses litert.sock before openai.sock.
            self.assertFalse(openai.pump(now=due))
            self.assertTrue(litert.pump(now=due))
            self.assertEqual(openai_sent, [])
            self.assertEqual(litert_sent, [(identifier, "push")])
            self.assertFalse(openai.pump(now=due + 1.0))

    def test_bound_sender_infers_core_socket_scope(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "outbox.sqlite3"
            openai_owner = Owner("/run/cat-agent/openai.sock", [], human=False)
            litert_owner = Owner("/run/cat-agent/litert.sock", [], human=False)
            openai = NotificationOutbox(db, openai_owner.send)
            litert = NotificationOutbox(db, litert_owner.send)

            self.assertEqual(openai.scope, "/run/cat-agent/openai.sock")
            self.assertEqual(litert.scope, "/run/cat-agent/litert.sock")


if __name__ == "__main__":
    unittest.main()
