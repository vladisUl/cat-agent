"""Real SQLite TASK SYSTEM + native worker, without a broker or model."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent_core_tools.task_worker import NativeTaskWorker
from litert_agent_tools.mqtt_tools import mqtt_event_catalog, create_mqtt_tools, _make_tool
from litert_agent_tools.mqtt_event_context import MQTT_EVENT_SNAPSHOT, MqttEventSnapshot
from orchestration.mqtt_events import MQTT_HOST, MQTT_PORT
from orchestration.mqtt_events import MqttEventMonitor
from orchestration.tasks import TaskStoreError
from orchestration_tools.delegation import DelegationBroker, create_delegate_tool
from orchestration_tools.task_tools import NativeTaskController
from task_system.engine import TaskEngine
from test_native_task_tools import LocalRPC, FakeManager


class Queue:
    def __init__(self):
        self.pending = []

    def submit(self, kind, callback):
        self.pending.append((kind, callback))


class NativeMqttEventsTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.db = self.root / 'tasks.sqlite3'
        self.engine = self.open_engine()
        self.addCleanup(lambda: self.engine.close())
        self.rpc = LocalRPC(self.engine)
        self.permissions = {'agent_1': ('get_corridor_motion', 'get_kitchen_leak')}
        self.controller = NativeTaskController(self.rpc, self.permissions, mqtt_event_catalog())
        self.controller.enabled = True
        self.broker = DelegationBroker(('agent_1',), self.controller)
        self.broker.enabled = True
        self.delegate = create_delegate_tool(self.broker)

    def open_engine(self):
        return TaskEngine(self.db, task_file=self.root / 'legacy.txt',
                          event_file=self.root / 'events.json', alive=lambda peer: True)

    def create(self, tool='get_corridor_motion', period=-1, field=''):
        self.controller.begin_turn()
        self.broker.begin_turn()
        try:
            return self.delegate('agent_1', 'Сообщи, если обнаружено движение; иначе молчи.',
                                 period, tool, field)
        finally:
            self.controller.finish_turn()
            self.broker.finish_turn()

    def test_event_to_native_agent_keeps_trigger_value_and_hardware_priority(self):
        result = self.create()
        self.assertTrue(result['ok'], result)
        identifier = result['task_id']
        task = self.engine.tasks.get(identifier)
        self.assertIsNone(task.timer_period_seconds)
        self.assertEqual((task.executor, task.agent_id), ('litert', 'agent_1'))
        self.assertFalse(self.engine.runtime.task_timer_snapshot())
        monitor = MqttEventMonitor(self.engine.events, self.engine.consume_mqtt)
        monitor._consume_line('zigbee2mqtt/dvigen_verh {"occupancy":false}')
        self.assertFalse(self.engine.runs)
        monitor._consume_line('zigbee2mqtt/dvigen_verh {"occupancy":true}')
        monitor._consume_line('zigbee2mqtt/dvigen_verh {"occupancy":true}')
        self.assertEqual(len(self.engine.runs), 1)
        # Sensor has already cleared by execution time: the true event must
        # remain true, and false is queued as its own independent transition.
        monitor._consume_line('zigbee2mqtt/dvigen_verh {"occupancy":false}')
        manager, queue = FakeManager(), Queue()
        sensor = next(t for t in create_mqtt_tools() if t.__name__ == 'get_corridor_motion')
        observed = []
        original_run = manager.run_agent
        def run_with_sensor(agent_id, instruction):
            with patch('litert_agent_tools.mqtt_tools._read_measurement',
                       side_effect=AssertionError('must not wait for a new MQTT message')):
                observed.append(sensor())
            return original_run(agent_id, instruction)
        manager.run_agent = run_with_sensor
        worker = NativeTaskWorker(queue, manager, rpc=self.rpc, agent_permissions=self.permissions)
        self.assertTrue(worker.poll_once())
        kind, callback = queue.pending.pop(0)
        self.assertEqual(kind, 'hardware')
        callback()
        worker.poll_once()  # persist first result and admit second event
        first = manager.calls[0][1]
        self.assertTrue(first.startswith(task.text))
        payload = json.loads(first.split('[SYSTEM_MQTT_EVENT]\n')[1].split('\n[/SYSTEM_MQTT_EVENT]')[0])
        self.assertEqual((payload['topic'], payload['field'], payload['value']),
                         ('zigbee2mqtt/dvigen_verh', 'occupancy', True))
        queue.pending.pop(0)[1]()
        worker.poll_once()
        self.assertIn('"value": false', manager.calls[1][1])
        self.assertEqual(len(worker.list_results()), 2)
        self.assertTrue(all(r['state'] == 'done' for r in self.engine.runs.values()))
        self.assertEqual([r['occupancy'] for r in observed], [True, False])
        self.assertTrue(all(r['ok'] and r['source'] == 'mqtt_event' for r in observed))
        self.assertIsNone(MQTT_EVENT_SNAPSHOT.get())
        with patch('litert_agent_tools.mqtt_tools._read_measurement',
                   return_value={'ok': True, 'occupancy': False}) as live:
            self.assertEqual(sensor(), {'ok': True, 'occupancy': False})
            live.assert_called_once()

    def test_event_snapshot_cleared_after_agent_failure(self):
        self.assertTrue(self.create()['ok'])
        self.engine.consume_mqtt(self.engine.events.snapshot()[0], 'true')
        manager, queue = FakeManager(), Queue()
        def fail(*args):
            self.assertIs(MQTT_EVENT_SNAPSHOT.get().value, True)
            raise RuntimeError('simulated model failure')
        manager.run_agent = fail
        worker = NativeTaskWorker(queue, manager, rpc=self.rpc, agent_permissions=self.permissions)
        self.assertTrue(worker.poll_once())
        with self.assertLogs('agent_core_tools.task_worker', level='ERROR'):
            queue.pending.pop(0)[1]()
        self.assertIsNone(MQTT_EVENT_SNAPSHOT.get())
        worker.poll_once()
        self.assertEqual(next(iter(self.engine.runs.values()))['state'], 'error')

    def test_snapshot_bypasses_cache_but_does_not_replace_other_sensor_fields(self):
        cache = Mock()
        snapshot = MqttEventSnapshot(MQTT_HOST, MQTT_PORT, 'motion', 'occupancy', False, 'task_mqtt1')
        token = MQTT_EVENT_SNAPSHOT.set(snapshot)
        try:
            sensor = _make_tool('motion', 'motion', (MQTT_HOST, MQTT_PORT, 5), 'motion',
                                {'occupancy': 'detected'}, {'occupancy': {'type': 'boolean'}}, cache)
            result = sensor()
            self.assertIs(result['detected'], False)
            self.assertEqual(result['source'], 'mqtt_event')
            cache.read.assert_not_called()
            for broker, topic, fields in (
                ((MQTT_HOST, MQTT_PORT, 5), 'leak', {'water_leak': 'leak'}),
                (('another-broker', MQTT_PORT, 5), 'motion', {'occupancy': 'occupancy'}),
                ((MQTT_HOST, MQTT_PORT, 5), 'motion', {'occupancy': 'occupancy', 'battery': 'battery'}),
            ):
                tool = _make_tool('other', 'other', broker, topic, fields, {})
                with patch('litert_agent_tools.mqtt_tools._read_measurement',
                           return_value={'ok': True, 'live': True}) as read:
                    self.assertEqual(tool(), {'ok': True, 'live': True})
                    read.assert_called_once()
        finally:
            MQTT_EVENT_SNAPSHOT.reset(token)

    def test_initial_active_state_and_persistent_binding(self):
        result = self.create('get_kitchen_leak', '-1.0')
        self.assertTrue(result['ok'], result)
        self.engine.close()
        self.engine = self.open_engine()
        root = Path(__file__).resolve().parents[1]
        monitor = MqttEventMonitor(self.engine.events, self.engine.consume_mqtt,
                                  active_state_path=root / 'config/mqtt_event_active.json')
        monitor._consume_line('zigbee2mqtt/kuhnya_leak {"water_leak":true}')
        self.assertEqual(len(self.engine.runs), 1)
        self.assertEqual(next(iter(self.engine.runs.values()))['event']['task'], 'true')

    def test_list_pause_resume_delete_external_task(self):
        result = self.create()
        ident = result['task_id']
        binding = self.engine.events.snapshot()[0]
        self.controller.begin_turn()
        try:
            self.assertEqual(self.controller.list_queries()['tasks'][0]['period_seconds'], -1)
            self.assertTrue(self.controller.stop(ident)['ok'])
            self.engine.consume_mqtt(binding, 'true')
            self.assertFalse(self.engine.runs)
            self.assertEqual(self.controller.start(ident)['period_seconds'], -1)
            self.assertFalse(self.engine.runtime.task_timer_snapshot())
            self.engine.consume_mqtt(binding, 'false')
            self.assertEqual(len(self.engine.runs), 1)
            self.assertTrue(self.controller.delete(ident)['ok'])
            self.assertFalse(self.engine.events.snapshot())
            self.engine.consume_mqtt(binding, 'true')
            self.assertEqual(len(self.engine.runs), 1)
        finally:
            self.controller.finish_turn()

    def test_invalid_sources_permissions_and_intervals_do_not_create_tasks(self):
        for tool, period, field in [('missing', -1, ''), ('get_corridor_motion', -1, 'wrong'),
                                    ('get_corridor_motion', -1.5, ''), ('get_corridor_motion', 60, '')]:
            self.assertFalse(self.create(tool, period, field)['ok'])
        self.controller.agent_permissions['agent_1'] = frozenset()
        self.assertFalse(self.create()['ok'])
        self.assertFalse(self.engine.tasks.list())
        self.assertFalse(self.engine.events.snapshot())

    def test_lost_create_reply_does_not_duplicate_binding(self):
        original = self.rpc.call
        def lost(*args, **kwargs):
            original(*args, **kwargs)
            raise TaskStoreError('Task SYSTEM unavailable: reply lost')
        self.controller.begin_turn()
        self.broker.begin_turn()
        try:
            with patch.object(self.rpc, 'call', side_effect=lost):
                first = self.delegate('agent_1', 'Leak alert', -1, 'get_kitchen_leak')
                second = self.delegate('agent_1', 'Leak alert', -1, 'get_kitchen_leak')
            self.assertEqual(first['status'], 'unknown')
            self.assertFalse(second['ok'])
            self.assertEqual(len(self.engine.events.snapshot()), 1)
            self.assertEqual(len(self.engine.tasks.list()), 1)
        finally:
            self.controller.finish_turn()
            self.broker.finish_turn()
