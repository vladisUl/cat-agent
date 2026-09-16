from pathlib import Path
import tempfile
import unittest

from task_system.client import RemoteSystemRuntime
from task_system.engine import TaskEngine
from test_task_system import LocalRPC


class AutoTaskTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = 100.0
        self.live = {'a', 'b'}
        self.engine = self.open_engine()
        self.addCleanup(lambda: self.engine.close())
        self.a = RemoteSystemRuntime('litert', rpc=LocalRPC(self.engine, 'a'))
        self.b = RemoteSystemRuntime('openai', rpc=LocalRPC(self.engine, 'b'))

    def open_engine(self):
        return TaskEngine(self.root / 'state.db', task_file=self.root / 'tasks.txt',
                          event_file=self.root / 'events.json', alive=lambda p: p in self.live,
                          clock=lambda: self.now, lease_seconds=20)

    def create(self, executor='auto'):
        task = self.a.create_periodic_task('check', 'read file', ('shell',), 60, executor=executor)
        self.engine.call('system', 'run_task', (task.task_id,))
        return task

    def pull(self, client):
        client._next_poll = 0
        return client.poll_due()

    def status(self, executor):
        return next(s for s in self.engine.core_status() if s['executor'] == executor)

    def test_both_ready_openai_wins_even_if_litert_polls_first(self):
        self.a.heartbeat(True)
        self.b.heartbeat(True)
        task = self.create()
        self.assertEqual(self.pull(self.a), ())
        event, = self.pull(self.b)
        self.assertEqual(self.engine.runs[event.run_id]['executor'], 'openai')
        self.assertEqual(self.a.task_store.require(task.task_id).executor, 'auto')

    def test_not_ready_openai_uses_litert(self):
        self.a.heartbeat(True)
        self.b.heartbeat(False, 'quota_exhausted')
        self.create()
        self.assertEqual(self.pull(self.b), ())
        event, = self.pull(self.a)
        self.assertEqual(self.engine.runs[event.run_id]['executor'], 'litert')
        self.assertEqual(self.status('openai')['reason'], 'quota_exhausted')

    def test_only_openai_ready(self):
        self.b.heartbeat(True)
        self.create()
        self.assertEqual(self.pull(self.a), ())
        self.assertEqual(len(self.pull(self.b)), 1)

    def test_both_not_ready_wait_and_recover(self):
        self.a.heartbeat(False, 'initializing')
        self.b.heartbeat(False, 'missing_api_key')
        self.create()
        self.assertEqual(self.pull(self.a), ())
        self.assertEqual(self.pull(self.b), ())
        run = next(iter(self.engine.runs.values()))
        self.assertEqual(run['state'], 'pending')
        self.assertIsNone(run['executor'])
        self.b.heartbeat(True)
        event, = self.pull(self.b)
        self.assertTrue(self.b.begin_run(event))

    def test_heartbeat_expiry_is_not_ready_even_if_process_alive(self):
        self.a.heartbeat(True)
        self.b.heartbeat(True)
        self.now += 20
        self.assertEqual(self.status('openai')['reason'], 'heartbeat_expired')
        self.create()
        self.assertEqual(self.pull(self.b), ())
        self.a.heartbeat(True)
        self.assertEqual(len(self.pull(self.a)), 1)

    def test_explicit_litert_is_not_taken_by_ready_openai(self):
        self.a.heartbeat(True)
        self.b.heartbeat(True)
        self.create('litert')
        self.assertEqual(self.pull(self.b), ())
        self.assertEqual(len(self.pull(self.a)), 1)

    def test_explicit_openai_works_without_quota_evidence(self):
        self.a.heartbeat(True)
        self.b.heartbeat(False, 'missing_api_key')
        self.create('openai')
        self.assertEqual(self.pull(self.a), ())
        event, = self.pull(self.b)
        self.assertTrue(self.b.begin_run(event))

    def test_offer_expiry_reassigns_and_fences_old_begin(self):
        self.a.heartbeat(True)
        self.b.heartbeat(True)
        self.create()
        old, = self.pull(self.b)
        self.b.accepted(old)
        self.now += 21
        self.a.heartbeat(True)
        new, = self.pull(self.a)
        self.assertEqual(new.run_id, old.run_id)
        self.assertNotEqual(new.offer_id, old.offer_id)
        self.assertFalse(self.b.begin_run(old))
        self.assertTrue(self.a.begin_run(new))
        self.b.finish_run(old, 'cancelled')
        self.b.flush_completions()
        self.assertEqual(self.engine.runs[new.run_id]['state'], 'running')

    def test_same_worker_new_grant_rejects_stale_begin_and_finish(self):
        self.b.heartbeat(True)
        self.create()
        old, = self.pull(self.b)
        self.b.accepted(old)
        self.now += 21
        self.b.heartbeat(True)
        new, = self.pull(self.b)
        self.assertNotEqual(new.offer_id, old.offer_id)
        self.assertFalse(self.b.begin_run(old))
        self.b.finish_run(old, 'cancelled')
        self.b.flush_completions()
        self.assertTrue(self.b.begin_run(new))

    def test_running_lease_expiry_does_not_replay_or_transfer(self):
        self.a.heartbeat(True)
        self.b.heartbeat(True)
        self.create()
        event, = self.pull(self.b)
        self.b.accepted(event)
        self.b.begin_run(event)
        self.now += 21
        self.a.heartbeat(True)
        self.assertEqual(self.status('openai')['state'], 'NOT_READY')
        self.assertEqual(self.pull(self.a), ())
        self.assertEqual(self.engine.runs[event.run_id]['state'], 'running')
        self.b.heartbeat(True)
        self.assertTrue(self.b.valid_run(event))

    def test_dead_core_before_begin_reassigns_after_begin_no_replay(self):
        self.a.heartbeat(True)
        self.b.heartbeat(True)
        self.create()
        event, = self.pull(self.b)
        self.live.remove('b')
        new, = self.pull(self.a)
        self.assertEqual(new.run_id, event.run_id)
        self.a.accepted(new)
        self.a.begin_run(new)
        self.live.remove('a')
        self.engine.tick()
        self.assertEqual(self.engine.runs[event.run_id]['state'], 'interrupted')
        self.live.add('b')
        self.b.heartbeat(True)
        self.assertEqual(self.pull(self.b), ())

    def test_ready_recovery_changes_next_run_not_task(self):
        self.a.heartbeat(True)
        self.b.heartbeat(False, 'usage_network_error')
        task = self.create()
        first, = self.pull(self.a)
        self.a.accepted(first)
        self.a.begin_run(first)
        self.b.heartbeat(True)
        self.assertEqual(self.pull(self.b), ())
        self.a.finish_run(first, 'done')
        self.a.flush_completions()
        self.engine.call('system', 'run_task', (task.task_id,))
        self.assertEqual(self.pull(self.a), ())
        second, = self.pull(self.b)
        self.assertNotEqual(first.run_id, second.run_id)
        self.assertEqual(self.a.task_store.require(task.task_id).executor, 'auto')

    def test_service_restart_does_not_restore_ready(self):
        self.b.heartbeat(True)
        task = self.create()
        self.engine.close()
        self.engine = self.open_engine()
        self.a.rpc.engine = self.b.rpc.engine = self.engine
        self.assertEqual(self.status('openai')['state'], 'NOT_READY')
        self.assertEqual(self.pull(self.b), ())
        self.b.heartbeat(True)
        self.assertEqual(len(self.pull(self.b)), 1)
        self.assertEqual(self.b.task_store.require(task.task_id).executor, 'auto')

    def test_new_defaults_auto_and_existing_pins_survive_reload(self):
        new = self.b.create_periodic_task('new', 'work', ('shell',), 60)
        pinned = self.create('litert')
        self.assertEqual(new.executor, 'auto')
        self.engine.close()
        self.engine = self.open_engine()
        self.assertEqual(self.engine.tasks.require(new.task_id).executor, 'auto')
        self.assertEqual(self.engine.tasks.require(pinned.task_id).executor, 'litert')

    def test_usage_api_failures_exclude_openai_from_auto(self):
        from unittest.mock import Mock, patch
        from urllib.error import HTTPError, URLError
        from task_system.ollama_usage import OllamaUsageProbe, USAGE_URL
        self.a.heartbeat(True)
        errors = [URLError('offline'), HTTPError(USAGE_URL, 503, 'error', {}, None),
                  HTTPError(USAGE_URL, 401, 'error', {}, None), HTTPError(USAGE_URL, 403, 'error', {}, None)]
        for error in errors:
            with self.subTest(error=type(error).__name__, code=getattr(error, 'code', None)):
                probe = OllamaUsageProbe(Mock(side_effect=error))
                with patch.dict('os.environ', {'OLLAMA_API_KEY': 'test-key'}):
                    status = probe.check()
                self.b.heartbeat(status.ready, status.reason)
                task = self.create()
                self.assertEqual(self.pull(self.b), ())
                event, = self.pull(self.a)
                self.assertEqual(self.engine.runs[event.run_id]['executor'], 'litert')
                self.a.delete_task(task.task_id)
