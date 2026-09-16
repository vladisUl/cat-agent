from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from orchestration.tasks import TaskStore, TaskStoreError
from task_system.client import RemoteSystemRuntime
from task_system.engine import TaskEngine


class LocalRPC:
    """Same JSON boundary as Unix RPC, without requiring sockets in the sandbox."""
    def __init__(self, engine, peer):
        self.engine, self.peer = engine, peer

    def call(self, target, operation, *args, **kwargs):
        result = self.engine.call(target, operation, args, kwargs, peer=self.peer)
        return json.loads(json.dumps(result, default=lambda x: asdict(x) if is_dataclass(x) else x))


class TaskSystemTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.live = {'a', 'b', 'c'}
        self.engine = self.open_engine()
        self.addCleanup(lambda: self.engine.close())
        self.a = RemoteSystemRuntime('litert', rpc=LocalRPC(self.engine, 'a'))
        self.b = RemoteSystemRuntime('openai', rpc=LocalRPC(self.engine, 'b'))
        self.c = RemoteSystemRuntime('litert', rpc=LocalRPC(self.engine, 'c'))

    def open_engine(self):
        return TaskEngine(self.root / 'system.db', task_file=self.root / 'tasks.txt',
                          event_file=self.root / 'events.json', alive=lambda peer: peer in self.live)

    def create(self, client=None, executor='litert'):
        return (client or self.a).create_periodic_task('test', 'do work', ('mqtt',), 10, executor=executor)

    def due(self, task):
        timer = self.engine.runtime._task_timers[task.task_id]
        self.engine.tick(timer.next_fire_monotonic + .01)

    def pull(self, client):
        client._next_poll = 0
        return client.poll_due()

    def test_shared_crud_and_default_executor(self):
        task = self.create(self.b)
        self.assertEqual(task.executor, 'litert')
        self.assertEqual(self.a.task_store.require(task.task_id), task)
        stopped = self.a.stop_task(task.task_id)
        self.assertFalse(self.b.task_store.require(task.task_id).enabled)
        self.assertNotEqual(stopped.generation, task.generation)
        self.b.set_task_period(task.task_id, 30)
        self.assertEqual(self.a.task_timer_snapshot()[0].period_seconds, 30)
        self.b.start_task(task.task_id)
        self.assertTrue(self.a.task_store.require(task.task_id).enabled)
        self.assertTrue(self.a.delete_task(task.task_id))
        self.assertEqual(self.b.task_snapshot(), ())
        self.assertEqual(self.b.task_timer_snapshot(), ())

    def test_offline_designated_executor_waits_without_fallback(self):
        task = self.create(executor='openai')
        self.due(task)
        self.assertEqual(self.pull(self.a), ())
        self.due(task)
        self.assertEqual(len(self.engine.runs), 1)
        event, = self.pull(self.b)
        self.b.accepted(event)
        self.assertTrue(self.b.begin_run(event))
        self.assertEqual(self.pull(self.b), ())
        self.assertEqual(self.pull(self.a), ())
        self.b.finish_run(event, 'silent')
        self.assertTrue(self.b.flush_completions())
        self.assertEqual(self.engine.runs[event.run_id]['state'], 'done')

    def test_two_cores_cannot_claim_same_task_concurrently(self):
        task = self.create()
        self.due(task)
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(self.pull, [self.a, self.c]))
        self.assertEqual(sum(len(r) for r in results), 1)

    def test_lost_offer_reply_returns_same_run(self):
        task = self.create()
        self.due(task)
        first, = self.pull(self.a)
        repeated, = self.pull(self.a)
        self.assertEqual(first.run_id, repeated.run_id)
        self.assertEqual(self.pull(self.c), ())

    def test_stop_delete_recreate_cannot_revive_active_work(self):
        task = self.create()
        self.due(task)
        event, = self.pull(self.a)
        self.a.accepted(event)
        self.assertTrue(self.a.begin_run(event))
        self.b.stop_task(task.task_id)
        self.assertFalse(self.a.valid_run(event))
        self.b.delete_task(task.task_id)
        replacement = self.create(self.b)
        self.assertEqual(replacement.task_id, task.task_id)
        self.due(replacement)
        self.assertEqual(self.pull(self.c), ())  # Old worker must first release execution.
        self.a.finish_run(event, 'cancelled')
        self.a.flush_completions()
        next_event, = self.pull(self.c)
        self.assertNotEqual(next_event.task_generation, event.task_generation)

    def test_service_restart_preserves_pending_and_running(self):
        task = self.create()
        self.due(task)
        event, = self.pull(self.a)
        self.a.accepted(event)
        self.assertTrue(self.a.begin_run(event))
        self.engine.close()
        self.engine = self.open_engine()
        for client in (self.a, self.b, self.c):
            client.rpc.engine = self.engine
        self.assertTrue(self.a.valid_run(event))
        self.assertEqual(self.pull(self.c), ())
        self.a.finish_run(event, 'done')
        self.a.flush_completions()
        self.due(self.engine.tasks.require(task.task_id))
        self.assertEqual(len(self.pull(self.c)), 1)

    def test_worker_death_requeues_only_unstarted_offer(self):
        task = self.create()
        self.due(task)
        event, = self.pull(self.a)
        self.live.remove('a')
        replacement, = self.pull(self.c)
        self.assertEqual(event.run_id, replacement.run_id)
        self.c.accepted(replacement)
        self.c.begin_run(replacement)
        self.live.remove('c')
        self.engine.tick()
        self.assertEqual(self.engine.runs[event.run_id]['state'], 'interrupted')
        self.live.add('a')
        self.assertEqual(self.pull(self.a), ())

    def test_executor_change_cancels_old_grant_waits_for_release(self):
        task = self.create()
        self.due(task)
        event, = self.pull(self.a)
        self.a.accepted(event)
        self.a.begin_run(event)
        updated = self.b.task_store.set_executor(task.task_id, 'openai')
        self.assertEqual(updated.executor, 'openai')
        self.assertFalse(self.a.valid_run(event))
        self.engine.call('system', 'run_task', (task.task_id,))
        self.assertEqual(self.pull(self.b), ())
        self.a.finish_run(event, 'cancelled')
        self.a.flush_completions()
        replacement, = self.pull(self.b)
        self.assertTrue(self.b.begin_run(replacement))

    def test_mqtt_registration_transaction_and_shared_binding(self):
        record = self.b.create_external_task(
            dict(description='mqtt', text='monitor', skills=['mqtt'], executor='openai'),
            dict(source='mqtt', topic='zigbee/x', field='state', value_type='str', values=['ON'], command='mqtt_sub.sh zigbee/x state'))
        name = f'task_mqtt{record.task_id}'
        self.assertIsNotNone(self.a.event_store.resolve('mqtt', name))
        self.engine.call('system', 'external_event', ('mqtt', name, 'ON'))
        self.assertEqual(self.pull(self.a), ())
        event, = self.pull(self.b)
        self.assertEqual(event.task, 'ON')
        self.a.delete_task(record.task_id)
        self.assertIsNone(self.b.event_store.resolve('mqtt', name))
        before = self.a.task_snapshot()
        with self.assertRaises(Exception):
            self.b.create_external_task(dict(description='bad', text='bad'), dict(source='mqtt'))
        self.assertEqual(self.a.task_snapshot(), before)

    def test_legacy_import_is_once_and_executor_is_persisted(self):
        self.engine.close()
        (self.root / 'system.db').unlink()
        old = TaskStore(self.root / 'tasks.txt')
        old.create('legacy', 'work', skills=('mqtt',), timer_period_seconds=60)
        legacy = json.loads((self.root / 'tasks.txt').read_text())
        legacy.pop('executor')
        (self.root / 'tasks.txt').write_text(json.dumps(legacy) + '\n')
        original = (self.root / 'tasks.txt').read_bytes()
        self.engine = self.open_engine()
        task = self.engine.tasks.require(1)
        self.assertEqual(task.executor, 'litert')
        self.engine.call('tasks', 'set_executor', (1, 'openai'))
        self.engine.close()
        self.engine = self.open_engine()
        self.assertEqual(self.engine.tasks.require(1).executor, 'openai')
        self.assertEqual((self.root / 'tasks.txt').read_bytes(), original)

    def test_second_service_cannot_open_same_database(self):
        with self.assertRaises(BlockingIOError):
            self.open_engine()

    def test_failed_commit_rolls_back_state(self):
        self.create()
        before = self.a.task_snapshot()
        with patch.object(self.engine, '_save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.a.stop_task(1)
        self.assertEqual(self.b.task_snapshot(), before)

    def test_duplicate_completion_does_not_restart_work(self):
        task = self.create()
        self.due(task)
        event, = self.pull(self.a)
        self.a.accepted(event)
        self.a.begin_run(event)
        self.a.finish_run(event, 'done')
        self.a.flush_completions()
        self.a.finish_run(event, 'done')
        self.a.flush_completions()
        self.assertEqual(self.engine.runs[event.run_id]['state'], 'done')
        self.assertEqual(self.pull(self.c), ())

    def test_invalid_executor_rejected(self):
        with self.assertRaises(TaskStoreError):
            self.create(executor='llama')
        self.assertEqual(self.a.task_snapshot(), ())

    def test_real_scheduler_obeys_stop_from_other_core_between_steps(self):
        from types import SimpleNamespace
        from agent_core.core_scheduler import CoreScheduler
        import test_assistant_manager as fixtures
        runtime, model = fixtures.AssistantManagerTest()._runtime(
            self.root, ['/work#printf first', '/work#touch forbidden'])
        runtime.system_runtime = self.a
        runtime.event_store = self.a.event_store
        task = self.b.create_periodic_task('test', 'test', ('shell',), 60, executor='litert')
        self.due(task)
        scheduler = CoreScheduler(SimpleNamespace(runtime=runtime, system_runtime=self.a))
        self.addCleanup(scheduler._executor.shutdown, wait=True)
        scheduler._poll_system_events()
        self.assertEqual(len(scheduler._pending), 1)
        scheduler._start_next()
        scheduler._active_future.result(timeout=1)
        scheduler._poll_future()
        self.assertEqual(len(model.calls), 1)
        self.b.stop_task(task.task_id)
        scheduler._start_next()
        scheduler._active_future.result(timeout=1)
        scheduler._poll_future()
        self.a.flush_completions()
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(next(iter(self.engine.runs.values()))['state'], 'cancelled')
        self.assertIsNotNone(runtime.pool.acquire())

    def test_pending_queue_and_deadline_survive_service_restart(self):
        task = self.create(executor='openai')
        self.due(task)
        deadline = self.engine.runtime._task_timers[task.task_id].next_fire_monotonic
        self.engine.close()
        self.engine = self.open_engine()
        self.b.rpc.engine = self.engine
        self.assertAlmostEqual(self.engine.runtime._task_timers[task.task_id].next_fire_monotonic, deadline, places=3)
        event, = self.pull(self.b)
        self.assertEqual(event.task_id, task.task_id)

    def test_read_operations_do_not_write_database(self):
        self.create()
        with patch.object(self.engine, '_save', side_effect=AssertionError('unexpected write')):
            self.b.task_snapshot()
            self.b.task_timer_snapshot()
            self.engine.tick()

    def test_shared_manager_accepts_explicit_executor(self):
        import test_assistant_manager as fixtures
        runtime, _ = fixtures.AssistantManagerTest()._runtime(self.root, [])
        runtime.system_runtime = self.b
        runtime.event_store = self.b.event_store
        result = runtime._execute_work_command('task_timer.sh 60 shell --executor openai -- "check file"')
        self.assertIn('SYSTEM_OK', result)
        self.assertEqual(self.a.task_store.require(1).executor, 'openai')
        runtime._execute_timer_command(['timer.sh', 'executor', 'litert', '1'])
        self.assertEqual(self.b.task_store.require(1).executor, 'litert')

    def test_retired_mqtt_binding_cannot_trigger_recreated_task(self):
        spec = dict(description='mqtt', text='monitor', skills=['mqtt'])
        binding = dict(source='mqtt', topic='zigbee/x', field='state', value_type='str', values=['ON'], command='mqtt_sub.sh zigbee/x state')
        self.b.create_external_task(spec, binding)
        old = self.engine.events.resolve('mqtt', 'task_mqtt1')
        self.a.delete_task(1)
        self.b.create_external_task(spec, binding)
        self.assertIsNone(self.engine.consume_mqtt(old, 'ON'))
        self.assertEqual(len(self.engine.runs), 0)
