from pathlib import Path
from unittest.mock import Mock
import sqlite3
import tempfile
import unittest

from agent_core.connection import ClientConnection
from agent_core.outbox import NotificationOutbox


class DeliveryTest(unittest.TestCase):
    def test_slow_client_queue_is_bounded_and_disconnects(self):
        sock = Mock()
        client = ClientConnection(sock, capacity=2)
        self.assertTrue(client.send({"type": "reply", "text": "a"}))
        self.assertTrue(client.send({"type": "reply", "text": "b"}))
        self.assertFalse(client.send({"type": "reply", "text": "c"}))
        self.assertTrue(client._closed)
        sock.close.assert_called_once()

    def test_status_is_replaced_but_reply_is_retained(self):
        client = ClientConnection(Mock(), capacity=2)
        client.send({"type": "status", "pending": 1})
        client.send({"type": "reply", "text": "answer"})
        client.send({"type": "status", "pending": 2})
        self.assertEqual(list(client._queue), [{"type": "reply", "text": "answer"}, {"type": "status", "pending": 2}])

    def test_notification_survives_restart_and_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "outbox.db"
            sender = Mock(side_effect=OSError("offline"))
            outbox = NotificationOutbox(path, sender)
            identifier = outbox.enqueue("important")
            self.assertTrue(outbox.pump(now=0))
            self.assertFalse(outbox.pump(now=1))
            delivered = Mock(return_value=True)
            recovered = NotificationOutbox(path, delivered)
            self.assertTrue(recovered.pump(now=11))
            delivered.assert_called_once_with(identifier, "important")
            self.assertFalse(recovered.pump(now=1000))

    def test_human_ack_prevents_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            outbox = NotificationOutbox(Path(temp) / "outbox.db", Mock(return_value=False))
            identifier = outbox.enqueue("notice")
            outbox.pump(now=0)
            outbox.acknowledge(identifier)
            self.assertFalse(outbox.pump(now=1000))
