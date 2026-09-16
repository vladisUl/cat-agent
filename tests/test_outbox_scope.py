from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from agent_core.outbox import NotificationOutbox


class NotificationOutboxScopeTest(unittest.TestCase):
    def test_shared_database_does_not_cross_pump_between_scopes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outbox.sqlite3"
            sent_openai: list[tuple[str, str]] = []
            sent_litert: list[tuple[str, str]] = []

            openai = NotificationOutbox(
                path,
                lambda identifier, text: sent_openai.append((identifier, text)) or True,
                scope="/run/cat-agent/openai.sock",
            )
            litert = NotificationOutbox(
                path,
                lambda identifier, text: sent_litert.append((identifier, text)) or True,
                scope="/run/cat-agent/litert.sock",
            )

            identifier = openai.enqueue("done")

            self.assertFalse(litert.pump())
            self.assertEqual(sent_litert, [])

            self.assertTrue(openai.pump())
            self.assertEqual(sent_openai, [(identifier, "done")])

    def test_bound_sender_infers_core_socket_scope(self) -> None:
        class Owner:
            def __init__(self, path: str) -> None:
                self.path = Path(path)

            def send(self, _identifier: str, _text: str) -> bool:
                return True

        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "outbox.sqlite3"
            openai_owner = Owner("/run/cat-agent/openai.sock")
            litert_owner = Owner("/run/cat-agent/litert.sock")
            openai = NotificationOutbox(db, openai_owner.send)
            litert = NotificationOutbox(db, litert_owner.send)

            self.assertEqual(openai.scope, "/run/cat-agent/openai.sock")
            self.assertEqual(litert.scope, "/run/cat-agent/litert.sock")


if __name__ == "__main__":
    unittest.main()
