import asyncio
from contextlib import asynccontextmanager
import importlib.util
import json
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orchestration.mcp_config import McpServerConfig
from orchestration.mcp_runtime import McpRuntime

HAS_EXTRA = all(importlib.util.find_spec(x) for x in ('anyio', 'jsonschema'))


def tool(name='echo', description='Echo text'):
    return SimpleNamespace(name=name, description=description,
        input_schema={'type':'object', 'properties':{'text':{'type':'string'}}, 'required':['text']})


class FakeResult:
    def model_dump(self, **kwargs):
        return {'content':[{'type':'text','text':'actual value'}], 'structuredContent':{'answer':42}, 'isError':False}


class FakeClient:
    def __init__(self, tools=None):
        self.tools = [tool()] if tools is None else tools
        self.server_capabilities = SimpleNamespace(tools=SimpleNamespace())
        self.session = self
        self.calls = []
        self.list_calls = []
        self.fail_call = False
        self.delay = 0

    async def list_tools(self, cursor=None):
        self.list_calls.append(cursor)
        index = int(cursor or 0)
        return SimpleNamespace(tools=self.tools[index:index+1],
            next_cursor=str(index+1) if index+1 < len(self.tools) else None)

    async def call_tool(self, name, arguments, **kwargs):
        self.calls.append((name, arguments))
        await asyncio.sleep(self.delay)
        if self.fail_call:
            self.fail_call = False
            raise ConnectionError('do not log secret')
        return FakeResult()


@unittest.skipUnless(HAS_EXTRA, 'install .[mcp] to test MCP connection lifecycle')
class McpRuntimeTest(unittest.TestCase):
    def runtime(self, clients, unavailable=None, **kwargs):
        unavailable = unavailable if unavailable is not None else set()
        @asynccontextmanager
        async def factory(config):
            if config.name in unavailable:
                raise ConnectionError('secret in URL')
            yield clients[config.name]
        configs = [McpServerConfig(name, True, 'stdio', 'unused', (),
                    connect_timeout_seconds=.2, call_timeout_seconds=.08, reconnect_delay_seconds=.03)
                   for name in clients]
        runtime = McpRuntime(configs, connection_factory=factory)
        self.addCleanup(runtime.close)
        runtime.start()
        return runtime

    def wait_ready(self, runtime, name):
        until = time.monotonic()+2
        while time.monotonic()<until:
            if runtime.snapshot()[name]['state']=='ready': return
            time.sleep(.01)
        self.fail(runtime.snapshot())

    def test_list_pages_namespaces_and_call(self):
        client = FakeClient([tool(),tool('other')])
        runtime = self.runtime({'one':client, 'two':FakeClient()})
        self.assertEqual([s.name for s in runtime.skills()], ['mcp:one:echo','mcp:one:other','mcp:two:echo'])
        self.assertEqual(client.list_calls, [None,'1'])
        response = runtime.call('mcp:one:echo', {'text':"O'Brien $(false)"})
        self.assertEqual(client.calls, [('echo',{'text':"O'Brien $(false)"})])
        self.assertEqual(json.loads(response.split('\n',1)[1])['structuredContent'], {'answer':42})
        self.assertIn('invalid_arguments', runtime.call('mcp:one:echo', {'text':123}))
        self.assertEqual(len(client.calls),1)

    def test_failed_server_does_not_break_other_server_and_catalog_stays_frozen(self):
        unavailable={'bad'}
        runtime = self.runtime({'good':FakeClient(),'bad':FakeClient()}, unavailable)
        frozen=runtime.skills()
        self.assertIn('actual value', runtime.call('mcp:good:echo', {'text':'a'}))
        self.assertEqual(runtime.snapshot()['bad']['state'],'unavailable')
        unavailable.clear()
        self.wait_ready(runtime,'bad')
        self.assertEqual(runtime.skills(),frozen)
        self.assertIn('unknown_tool',runtime.call('mcp:bad:echo', {'text':'a'}))

    def test_reconnect_keeps_catalog_and_does_not_replay(self):
        client=FakeClient()
        runtime=self.runtime({'local':client})
        frozen=runtime.skills()
        client.fail_call=True
        self.assertIn('outcome_unknown',runtime.call('mcp:local:echo',{'text':'a'}))
        self.wait_ready(runtime,'local')
        self.assertEqual(runtime.skills(),frozen)
        self.assertEqual(len(client.calls),1)
        self.assertIn('actual value',runtime.call('mcp:local:echo',{'text':'b'}))
        self.assertEqual(len(client.calls),2)

    def test_timeout_and_shutdown(self):
        client=FakeClient()
        runtime=self.runtime({'local':client})
        client.delay=.2
        self.assertIn('outcome_unknown',runtime.call('mcp:local:echo',{'text':'a'}))
        self.wait_ready(runtime,'local')
        self.assertEqual(len(client.calls),1)
        runtime.close()
        self.assertFalse(runtime._thread.is_alive())
        self.assertIn('unavailable',runtime.call('mcp:local:echo',{'text':'b'}))

    def test_duplicate_tool_names_disable_only_that_server(self):
        runtime=self.runtime({'bad':FakeClient([tool(),tool()]),'good':FakeClient()})
        self.assertEqual([s.name for s in runtime.skills()],['mcp:good:echo'])
        self.assertEqual(runtime.snapshot()['bad']['state'],'unavailable')

    def test_changed_schema_does_not_mutate_catalog_or_send_call(self):
        client=FakeClient()
        runtime=self.runtime({'local':client})
        before=runtime.skills()
        client.tools=[tool('replacement')]
        client.fail_call=True
        runtime.call('mcp:local:echo',{'text':'a'})
        self.wait_ready(runtime,'local')
        self.assertIn('tool_changed',runtime.call('mcp:local:echo',{'text':'b'}))
        self.assertEqual(runtime.skills(),before)
        self.assertEqual(len(client.calls),1)

    def test_missing_sdk_is_nonfatal(self):
        @asynccontextmanager
        async def missing(config):
            raise ImportError('mcp missing')
            yield
        runtime=McpRuntime([McpServerConfig('missing',True,'stdio','unused')],connection_factory=missing)
        self.addCleanup(runtime.close)
        runtime.start()
        self.assertEqual(runtime.skills(),())
        self.assertEqual(runtime.snapshot()['missing']['state'],'unavailable')

    def test_server_without_tools_capability(self):
        client=FakeClient()
        client.server_capabilities.tools=None
        runtime=self.runtime({'local':client})
        self.assertEqual(runtime.skills(),())
        self.assertEqual(client.list_calls,[])
        self.assertEqual(runtime.snapshot()['local']['state'],'ready')

    def test_connection_timeout_does_not_block_core_or_other_server(self):
        client=FakeClient()
        @asynccontextmanager
        async def factory(config):
            if config.name=='slow':
                await asyncio.sleep(5)
            yield client
        configs=[McpServerConfig(name,True,'stdio','unused',connect_timeout_seconds=.04,
                                reconnect_delay_seconds=.03) for name in ('slow','good')]
        runtime=McpRuntime(configs,connection_factory=factory)
        self.addCleanup(runtime.close)
        before=time.monotonic()
        runtime.start()
        self.assertLess(time.monotonic()-before,2)
        self.assertEqual(runtime.snapshot()['slow']['state'],'unavailable')
        self.assertEqual([s.name for s in runtime.skills()],['mcp:good:echo'])
        self.assertIn('actual value',runtime.call('mcp:good:echo',{'text':'x'}))
