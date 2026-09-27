"""Exercise the real tool/manager/agent generators, without model hardware."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import unittest

from agent_core.core_scheduler import CoreScheduler, _PriorityRequest
from agent_core.voice_scheduler import VoiceCoreScheduler
from agent_core.types import InferenceTiming
from orchestration.cyclic_process import execute_cyclic_process
from orchestration.fragment_processing import fragment_observer
from orchestration.read_partly import read_partly_steps
from orchestration.skill_script import skill_script_steps
from orchestration.system_events import SystemEvent
from orchestration.workspace_command_runtime import CommandRuntime
import test_assistant_manager as manager_fixture
from test_cyclic_process import FakeClient


class FragmentSchedulingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def files(self, workspace):
        data = workspace / 'data'
        data.mkdir(exist_ok=True)
        (data / 'story.txt').write_text('AAA\nBBB\n', encoding='utf-8')
        (data / '1story.txt').write_text('AAA\n', encoding='utf-8')
        (data / '2story.txt').write_text('BBB\n', encoding='utf-8')
        (workspace / 'cyclic_process.sh').write_text('cyclic_process\n', encoding='utf-8')
        return data

    def setup_tool(self):
        self.files(self.root)
        runtime = CommandRuntime(self.root, ('read_partly', 'cyclic_process'),
                                 max_file_bytes=4096, timeout_seconds=2)
        client = FakeClient([], ['one', 'two'])
        return runtime, client

    def test_read_partly_yields_and_close_stops_before_next_fragment(self):
        runtime, client = self.setup_tool()
        seen = []
        token = fragment_observer.set(seen.append)
        self.addCleanup(fragment_observer.reset, token)
        steps = read_partly_steps('read_partly.sh story.txt -n 2', runtime, client,
                                  task_text='summarize')
        next(steps)
        child = client.children[0]
        self.assertEqual(len(child.calls), 1)
        self.assertEqual(len(child.reset_calls), 1)
        self.assertEqual(seen[-1]['phase'], 'between_fragments')
        self.assertEqual((seen[-1]['index'], seen[-1]['total']), (1, 2))
        steps.close()
        self.assertTrue(child.closed)
        self.assertEqual(len(child.calls), 1)
        self.assertIsNone(seen[-1])

    def test_cancel_scenario_keeps_previous_output_and_removes_temp_file(self):
        runtime, client = self.setup_tool()
        output = self.root / 'data/story_out.txt'
        output.write_text('previous', encoding='utf-8')
        steps = skill_script_steps('cyclic_process.sh story.txt -n 2', runtime, client,
                                   task_text='summarize')
        next(steps)
        self.assertEqual(len(client.children[0].calls), 1)
        steps.close()
        self.assertTrue(client.children[0].closed)
        self.assertEqual(output.read_text(), 'previous')
        self.assertEqual(list(output.parent.glob('*.tmp')), [])

    def test_mkstemp_failure_closes_child_and_returns_tool_error(self):
        runtime, client = self.setup_tool()
        with patch('orchestration.cyclic_process.tempfile.mkstemp', side_effect=OSError('disk full')):
            with self.assertLogs('orchestration.cyclic_process', level='ERROR'):
                result = execute_cyclic_process('cyclic_process story.txt -n 2', runtime, client,
                                                task_text='summarize')
        self.assertFalse(result.ok)
        self.assertIn('disk full', result.stderr)
        self.assertTrue(client.children[0].closed)
        self.assertEqual(client.children[0].calls, [])

    def test_unlink_failure_does_not_skip_child_close(self):
        runtime, client = self.setup_tool()
        steps = skill_script_steps('cyclic_process.sh story.txt -n 2', runtime, client,
                                   task_text='summarize')
        next(steps)
        with patch.object(Path, 'unlink', side_effect=OSError('cannot unlink')):
            with self.assertLogs('orchestration.cyclic_process', level='ERROR'):
                steps.close()
        self.assertTrue(client.children[0].closed)

    def test_task_stop_or_delete_between_fragments_releases_real_agent(self):
        for tool in ('read_partly', 'cyclic_process'):
            for operation in ('stop_task', 'delete_task'):
                with self.subTest(tool=tool, operation=operation), tempfile.TemporaryDirectory() as temp:
                    runtime, client = manager_fixture.AssistantManagerTest()._runtime(
                        Path(temp), [f'/work#{tool}.sh story.txt -n 2'])
                    self.files(runtime._direct_runtime.root)
                    child = FakeClient(['one', 'two'])
                    client.fork = Mock(return_value=child)
                    task = runtime.system_runtime.create_periodic_task('test', 'test', (tool,), 60, method='query')
                    execution = runtime.begin_autonomous_task(
                        SystemEvent('timer', 'task:1', '', 0, task.task_id, task.generation))
                    self.assertIsNone(runtime.step_autonomous_task(execution))
                    self.assertEqual(len(child.calls), 1)
                    getattr(runtime.system_runtime, operation)(task.task_id)
                    completion = runtime.step_autonomous_task(execution)
                    self.assertEqual(completion.turn.kind, 'cancelled')
                    self.assertEqual(execution.worker.state.value, 'FREE')
                    self.assertTrue(child.closed)
                    self.assertEqual(len(child.calls), 1)

    def test_scheduler_checks_run_validity_at_fragment_boundary_without_replay(self):
        runtime, client = manager_fixture.AssistantManagerTest()._runtime(
            self.root, ['/work#read_partly.sh story.txt -n 2'])
        self.files(runtime._direct_runtime.root)
        child = FakeClient(['one', 'two'])
        client.fork = Mock(return_value=child)
        task = runtime.system_runtime.create_periodic_task('test', 'test', ('read_partly',), 60, method='query')
        event = SystemEvent('timer', 'task:1', '', 0, task.task_id, task.generation)
        system = SimpleNamespace(begin_run=Mock(return_value=True), valid_run=Mock(return_value=False))
        scheduler = CoreScheduler(SimpleNamespace(runtime=runtime, system_runtime=system))
        self.addCleanup(scheduler._executor.shutdown, wait=True)
        item = _PriorityRequest('system', 'task:1', event, 0, 100, task_run=event)
        self.assertIsNone(scheduler._advance(item))
        self.assertEqual(len(child.calls), 1)
        self.assertEqual(item.fragment_progress['index'], 1)
        self.assertEqual(scheduler._advance(item).kind, 'cancelled')
        self.assertIsNone(item.fragment_progress)
        self.assertTrue(child.closed)
        self.assertEqual(len(child.calls), 1)
        system.begin_run.assert_called_once()
        system.valid_run.assert_called_once()

    def test_voice_preempts_real_manager_fragment_tool_and_result_is_preserved(self):
        for tool in ('read_partly', 'cyclic_process'):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as temp:
                runtime, client = manager_fixture.AssistantManagerTest()._runtime(
                    Path(temp), [f'/work#{tool}.sh story.txt -n 2', 'REPLY\nfinished'])
                data = self.files(runtime._direct_runtime.root)
                child = FakeClient(['one', 'two'])
                voice = manager_fixture.FakeClient(['REPLY\nvoice'])
                voice.close = Mock()
                client.fork = lambda label, **kw: voice if label == 'manager' else child
                delivered = []
                scheduler = VoiceCoreScheduler(SimpleNamespace(runtime=runtime),
                    on_completed=lambda req, turn: delivered.append((req.label, turn.text)))
                self.addCleanup(scheduler._executor.shutdown, wait=True)
                self.addCleanup(scheduler._close_all_contexts)
                scheduler.submit_user('summarize', session_id='web')
                def advance():
                    scheduler._start_next()
                    self.assertIsNotNone(scheduler._active_future)
                    scheduler._active_future.result(timeout=2)
                    scheduler._poll_future()
                advance()  # manager asks for the tool
                advance()  # first fragment, then suspension
                self.assertEqual(len(child.calls), 1)
                self.assertEqual(scheduler._pending[0].fragment_progress['total'], 2)
                scheduler.submit_voice('hello', session_id='mic')
                advance()
                advance()
                self.assertEqual(delivered, [('voice', 'voice')])
                self.assertEqual(len(child.calls), 1)
                for _ in range(12):
                    if not scheduler._pending:
                        break
                    advance()
                self.assertEqual(delivered, [('voice', 'voice'), ('user', 'finished')])
                self.assertTrue(child.closed)
                self.assertEqual(len(child.calls), 2)
                result = client.calls[-1][-1]['content']
                if tool == 'read_partly':
                    self.assertEqual(result, 'one\ntwo')
                else:
                    self.assertEqual(result, 'story_out.txt')
                    self.assertEqual((data / result).read_text(), 'one\ntwo\n')

    def test_snapshot_shows_suspended_fragment_and_cancel_pending(self):
        client = SimpleNamespace(resident_tokens=0,
            inference_timing=InferenceTiming('idle', None, None, None, None, None))
        scheduler = CoreScheduler(SimpleNamespace(manager_client=client, agent_client=client))
        self.addCleanup(scheduler._executor.shutdown, wait=True)
        item = _PriorityRequest('user', 'user', '', 0, 0, cancelled=True,
            fragment_progress={'tool':'read_partly', 'index':1, 'total':2, 'phase':'between_fragments'})
        scheduler._pending.append(item)
        progress = scheduler.status_snapshot()['fragment_runs']
        self.assertEqual(progress[0]['index'], 1)
        self.assertTrue(progress[0]['cancel_pending'])
