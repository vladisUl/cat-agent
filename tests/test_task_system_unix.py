from pathlib import Path
import socket
import tempfile
import threading
import unittest

from task_system.client import RemoteSystemRuntime, TaskRPC
from task_system.engine import TaskEngine
from task_system.main import Server, Handler


class TaskSystemUnixTest(unittest.TestCase):
    def test_two_clients_use_real_rpc_with_designated_executor(self):
        try:
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        except PermissionError:
            self.skipTest('environment forbids Unix sockets; run on Radxa')
        probe.close()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = TaskEngine(root / 'state.db', task_file=root / 'tasks.txt', event_file=root / 'events.json')
            try:
                with Server(str(root / 'tasks.sock'), Handler) as server:
                    server.engine = engine
                    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02})
                    thread.start()
                    try:
                        rpc = TaskRPC(root / 'tasks.sock')
                        a = RemoteSystemRuntime('litert', rpc=rpc)
                        b = RemoteSystemRuntime('openai', rpc=rpc)
                        task = b.create_periodic_task('check', 'read file', ('shell',), 60,
                                                      method='query', executor='litert')
                        self.assertEqual(a.task_store.require(task.task_id), task)
                        rpc.call('system', 'run_task', task.task_id)
                        self.assertEqual(b.poll_due(), ())
                        event, = a.poll_due()
                        a.accepted(event)
                        self.assertTrue(a.begin_run(event))
                        b.stop_task(task.task_id)
                        self.assertFalse(a.valid_run(event))
                        a.finish_run(event, 'cancelled')
                        self.assertTrue(a.flush_completions())
                        self.assertEqual(rpc.call('system', 'runs')[0]['state'], 'cancelled')
                        b.delete_task(task.task_id)
                        self.assertEqual(a.task_snapshot(), ())
                    finally:
                        server.shutdown()
                        thread.join()
            finally:
                engine.close()
