"""Native offline delivery reuses the shared persistent notification queue."""
from pathlib import Path
import queue
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import Mock

from agent_core_tools.server import CoreServer
from agent_core_tools.native_terminal import NativeTerminalTUI
from agent_core.outbox import NotificationOutbox

NOTICE = dict(type='task_query_result', run_id='run-1', agent_id='agent_1',
              task_id=1, text='Движение обнаружено')


class NativeNotificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'outbox.sqlite3'
        self.server = self.make_server()
        self.worker = Mock()
        self.worker.list_results.return_value = [dict(NOTICE)]
        self.server._task_worker = self.worker

    def make_server(self):
        server = CoreServer(Path(self.temp.name) / 'native.sock', outbox_path=self.path)
        server._firebase_sender = Mock()
        self.addCleanup(server.outbox.close)
        return server

    def row(self):
        with sqlite3.connect(self.path) as db:
            return db.execute('SELECT delivered, attempts, error FROM notifications').fetchone()

    def test_offline_handoff_and_repeated_run_push_only_once(self):
        self.server._publish_task_query_result(dict(NOTICE))
        self.worker.ack_result.assert_called_once_with('run-1')
        self.assertIsNone(self.row()[0])
        self.server.outbox.pump()
        self.server._publish_task_query_result(dict(NOTICE))
        self.server.outbox.pump()
        self.server._firebase_sender.send.assert_called_once_with(NOTICE['text'])
        self.assertIsNotNone(self.row()[0])

    def test_failed_handoff_does_not_ack_task_inbox(self):
        self.server.outbox.enqueue = Mock(side_effect=OSError('disk full'))
        with self.assertRaises(OSError):
            self.server._publish_task_query_result(dict(NOTICE))
        self.worker.ack_result.assert_not_called()

    def test_failed_push_survives_restart(self):
        self.server._firebase_sender.send.side_effect = RuntimeError('offline')
        self.server._publish_task_query_result(dict(NOTICE))
        now = time.time()
        self.server.outbox.pump(now)
        self.assertEqual(self.row(), (None, 1, 'offline'))
        self.server.outbox.close()
        restarted = self.make_server()
        restarted.outbox.pump(now + 11)
        restarted._firebase_sender.send.assert_called_once_with(NOTICE['text'])
        self.assertIsNotNone(self.row()[0])

    def test_live_result_then_disconnect_before_ack_is_recovered(self):
        client = Mock(pending_notifications={})
        self.server._human = client
        self.server._publish_task_query_result(dict(NOTICE))
        client.send.assert_called_once_with(NOTICE)
        self.worker.ack_result.assert_not_called()
        self.server._recover_offline_queries()
        self.worker.list_results.assert_not_called()
        self.server._human = None
        self.server._recover_offline_queries()
        self.server.outbox.pump()
        self.server._firebase_sender.send.assert_called_once_with(NOTICE['text'])

    def test_lost_task_ack_recovery_is_idempotent(self):
        self.worker.ack_result.side_effect = [OSError('lost ACK'), True]
        with self.assertRaises(OSError):
            self.server._publish_task_query_result(dict(NOTICE))
        self.server.outbox.pump()
        self.server._recover_offline_queries()
        self.server.outbox.pump()
        self.server._firebase_sender.send.assert_called_once()

    def test_reconnected_native_interface_acknowledges_outbox(self):
        self.server._publish_task_query_result(dict(NOTICE))
        client = Mock()
        self.server._human = client
        self.server.outbox.pump()
        frame = client.send.call_args.args[0]
        self.assertEqual(frame['type'], 'notification')
        self.assertIsNone(self.row()[0])
        self.server._handle_message(client, dict(type='notification_ack',
                                                notification_id=frame['notification_id']))
        self.assertIsNotNone(self.row()[0])
        self.server._firebase_sender.send.assert_not_called()

    def test_other_core_interface_takes_precedence_over_push(self):
        class Owner:
            _human = object()
            path = Path('/other.sock')
            def send(self, identifier, text):
                received.append((identifier, text))
                return False
        received = []
        other = NotificationOutbox(self.path, Owner().send)
        self.addCleanup(other.close)
        other.pump()
        self.server._publish_task_query_result(dict(NOTICE))
        self.server.outbox.pump()
        other.pump()
        self.assertEqual(received, [('native-query:run-1', NOTICE['text'])])
        self.server._firebase_sender.send.assert_not_called()

    def test_terminal_displays_once_and_retries_outbox_ack(self):
        client = Mock(incoming=queue.SimpleQueue())
        ui = NativeTerminalTUI(client, {}, {})
        frame = dict(type='notification', notification_id='native-query:run-1', text='done')
        def send(message):
            self.assertEqual(len(ui._dialog), 1)
            self.assertEqual(message['type'], 'notification_ack')
        client.send.side_effect = OSError('disconnected')
        client.incoming.put(frame)
        ui._poll_core()
        client.send.side_effect = send
        client.incoming.put(frame)
        ui._poll_core()
        self.assertEqual(len(ui._dialog), 1)
        self.assertEqual(client.send.call_count, 2)
