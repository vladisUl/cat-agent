"""Contract tests for LiteRT's public automatic-calling hooks (0.18/0.19)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from litert_agent_tools.conversation import NativeConversationChat, ToolExecutionLedger
from litert_agent_tools.tool_guard import ResultStore, ToolGuard, ToolTurnStopped, TURN_BUDGET, wire_size
from orchestration_tools.delegation import create_delegate_tool, DelegationBroker
from orchestration_tools.task_tools import create_task_tools, NativeTaskController
from orchestration_tools.agent_config import AgentSpec, effective_agent_tool_names
from test_litert_agent_tools import StubEngine


def sdk_dispatch(guard, tools, name, arguments=None):
    # The upstream SDK calls these two hooks OUTSIDE its execute try/except.
    # This placement is essential: throwing inside a tool alone cannot stop it.
    call = {'function': {'name': name, 'arguments': arguments or {}}}
    if not guard.approve_tool_call(call):
        return None
    if name not in tools:
        result = f'Error: interfaces.Tool {name} not found'
    else:
        try:
            result = tools[name](**call['function']['arguments'])
        except Exception as exc:
            result = f'Error: {exc}'
    return guard.process_tool_response(result)


class GuardTests(unittest.TestCase):
    def guard(self, **kwargs):
        ledger = ToolExecutionLedger()
        ledger.active = True
        guard = ToolGuard(ledger, **kwargs)
        guard.conversation = SimpleNamespace(token_count=100)
        return guard

    def test_public_signatures_have_real_integer_types(self):
        tools = [create_delegate_tool(DelegationBroker(('agent_1',))),
                 *create_task_tools(NativeTaskController(None, {'agent_1': ()}))]
        for tool in tools:
            wrapped = ToolExecutionLedger().wrap(tool)
            parameters = inspect.signature(wrapped).parameters
            for key in ('period_seconds', 'task_id'):
                if key in parameters:
                    self.assertIs(parameters[key].annotation, int)
        self.assertEqual(inspect.signature(ToolExecutionLedger().wrap(tools[0])).parameters['period_seconds'].default, 0)

    def test_repeated_failures_stop_sdk_loop_without_replay(self):
        guard = self.guard(max_calls=3)
        action = Mock(side_effect=RuntimeError('failed action'))
        wrapped = guard.ledger.wrap(lambda: action())
        with self.assertRaisesRegex(ToolTurnStopped, 'call limit'):
            for _ in range(100):
                sdk_dispatch(guard, {'action': wrapped}, 'action')
        self.assertEqual(action.call_count, 3)
        self.assertEqual(guard.ledger.calls, 3)
        self.assertEqual(guard.ledger.status_code, 1)

    def test_unknown_tool_is_failed_and_counts_against_limit(self):
        guard = self.guard(max_calls=1)
        sdk_dispatch(guard, {}, 'invented')
        self.assertEqual((guard.ledger.calls, guard.ledger.status_code), (1, 1))
        with self.assertRaises(ToolTurnStopped):
            sdk_dispatch(guard, {}, 'invented')

    def test_success_counted_once_and_failure_status_preserved(self):
        guard = self.guard()
        tool = guard.ledger.wrap(lambda: {'ok': True, 'value': 24.68})
        self.assertEqual(sdk_dispatch(guard, {'read': tool}, 'read')['value'], 24.68)
        self.assertEqual((guard.ledger.calls, guard.ledger.status_code), (1, 0))
        sdk_dispatch(guard, {}, 'invented')
        sdk_dispatch(guard, {'read': tool}, 'read')
        self.assertEqual((guard.ledger.calls, guard.ledger.status_code), (3, 1))

    def test_cancellation_and_deadline_prevent_next_side_effect(self):
        for mode in ('cancel', 'deadline'):
            guard = self.guard()
            action = Mock(return_value={'ok': True})
            if mode == 'cancel':
                guard.reset(lambda: False)
            else:
                guard.deadline = 0
            with self.assertRaises(ToolTurnStopped):
                sdk_dispatch(guard, {'action': action}, 'action')
            action.assert_not_called()

    def test_cancel_during_tool_does_not_run_another_generation(self):
        guard = self.guard()
        alive = [True]
        guard.reset(lambda: alive[0])
        def action():
            alive[0] = False
            return {'ok': True}
        with self.assertRaisesRegex(ToolTurnStopped, 'after tool execution'):
            sdk_dispatch(guard, {'action': action}, 'action')

    def test_budget_shared_across_manager_agent_and_report(self):
        token = TURN_BUDGET.set({})
        try:
            manager, agent = self.guard(max_calls=2), self.guard(max_calls=2)
            sdk_dispatch(manager, {'delegate': lambda: {'ok': True}}, 'delegate')
            sdk_dispatch(agent, {'read': lambda: {'ok': True}}, 'read')
            manager.reset()
            with self.assertRaisesRegex(ToolTurnStopped, 'call limit'):
                sdk_dispatch(manager, {}, 'third')
        finally:
            TURN_BUDGET.reset(token)

    def test_large_unicode_result_saved_and_read_without_repeating_tool(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ResultStore(Path(temp))
            guard = self.guard(store=store, max_result_bytes=512)
            original = {'ok': False, 'exit_code': 5, 'output': 'Температура 🐈\n' * 1500}
            action = Mock(return_value=original)
            summary = sdk_dispatch(guard, {'read': action}, 'read')
            self.assertTrue(summary['truncated'])
            self.assertFalse(summary['ok'])
            self.assertEqual(summary['exit_code'], 5)
            self.assertLessEqual(wire_size(summary), 512)
            identifier = summary['result_id']
            self.assertEqual(json.loads((Path(temp) / (identifier + '.json')).read_text()), original)
            reader = store.tool()
            parts, offset = [], 0
            while offset is not None:
                page = reader(identifier, offset, 512)
                self.assertTrue(page['ok'])
                parts.append(page['text'])
                offset = page['next_offset']
            self.assertEqual(json.loads(''.join(parts)), original)
            action.assert_called_once()
            self.assertFalse(reader('../secret')['ok'])

    def test_batch_results_share_remaining_context(self):
        guard = self.guard(max_context_tokens=1600, max_output_tokens=512, max_result_bytes=2048)
        # 732 bytes remain; several results before another native prefill
        # must not each get that entire budget.
        sdk_dispatch(guard, {'read': lambda: 'x' * 400}, 'read')
        action = Mock(return_value='next')
        with self.assertRaisesRegex(ToolTurnStopped, 'context'):
            sdk_dispatch(guard, {'read': action}, 'read')
        action.assert_not_called()

    def test_each_clone_has_guard_bound_to_its_own_kv(self):
        with NativeConversationChat(StubEngine(), 'system', []) as chat:
            first = chat._working
            chat.new_dialogue()
            second = chat._working
            self.assertIs(first.tool_event_handler.conversation, first)
            self.assertIs(second.tool_event_handler.conversation, second)
            self.assertIsNot(first.tool_event_handler, second.tool_event_handler)

    def test_agent_can_opt_out_of_shared_catalog(self):
        self.assertEqual(effective_agent_tool_names(
            AgentSpec('agent_1', Path('prompt'), ('only_this',), False), ('other',)), ('only_this',))


if __name__ == '__main__':
    unittest.main()
