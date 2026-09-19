"""Real SDK/wire tests. No network, model or external MCP account required for stdio."""
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from orchestration.mcp_config import McpServerConfig
from orchestration.mcp_runtime import McpRuntime

ROOT=Path(__file__).resolve().parents[1]
SERVER=ROOT/'tests/fixtures/mcp_server.py'
HAS_SDK=importlib.util.find_spec('mcp') is not None


@unittest.skipUnless(HAS_SDK, 'install .[mcp] for official SDK integration tests')
class McpSdkIntegrationTest(unittest.TestCase):
    def connect(self,config):
        runtime=McpRuntime([config])
        self.addCleanup(runtime.close)
        runtime.start()
        self.assertEqual(runtime.snapshot()[config.name]['state'],'ready',runtime.snapshot())
        return runtime

    def test_stdio_discovery_call_tool_error_and_process_cleanup(self):
        runtime=self.connect(McpServerConfig('local',True,'stdio',sys.executable,(str(SERVER),)))
        self.assertIn('mcp:local:echo',[s.name for s in runtime.skills()])
        result=runtime.call('mcp:local:echo',{'text':"O'Brien real stdio"})
        self.assertIn("O'Brien real stdio",result)
        error=runtime.call('mcp:local:fail',{})
        self.assertTrue(json.loads(error.split('\n',1)[1])['isError'])
        self.assertEqual(runtime.snapshot()['local']['state'],'ready')
        identity=json.loads(runtime.call('mcp:local:process_id',{}).split('\n',1)[1])
        pid=int(identity['content'][0]['text'])
        runtime.close()
        self.assertFalse(runtime._thread.is_alive())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid,0)

    def test_stdio_timeout_reconnect_without_replaying_side_effect(self):
        runtime=self.connect(McpServerConfig('local',True,'stdio',sys.executable,(str(SERVER),),
            connect_timeout_seconds=10,call_timeout_seconds=1,reconnect_delay_seconds=.05))
        before=runtime.skills()
        with tempfile.TemporaryDirectory() as temp:
            marker=Path(temp)/'calls.txt'
            result=runtime.call('mcp:local:slow',{'marker':str(marker)})
            self.assertIn('outcome_unknown',result)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                if runtime.snapshot()['local']['state']=='ready':
                    break
                time.sleep(.02)
            self.assertEqual(runtime.snapshot()['local']['state'],'ready')
            self.assertEqual(marker.read_text(),'called\n')
            self.assertEqual(runtime.skills(),before)
            self.assertIn('after reconnect',runtime.call('mcp:local:echo',{'text':'after reconnect'}))

    def test_streamable_http_real_socket(self):
        try:
            listener=socket.socket()
            listener.bind(('127.0.0.1',0))
        except PermissionError:
            self.skipTest('environment forbids local TCP sockets; run on Radxa')
        port=listener.getsockname()[1]
        listener.close()
        process=subprocess.Popen([sys.executable,str(SERVER),'--port',str(port)],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        def cleanup():
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        self.addCleanup(cleanup)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.1): break
            except OSError:
                if process.poll() is not None: self.fail('HTTP MCP server exited')
                time.sleep(.02)
        clean_proxy=patch.dict(os.environ, {key: '' for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy')})
        clean_proxy.start()
        self.addCleanup(clean_proxy.stop)
        runtime=self.connect(McpServerConfig('remote',True,'streamable_http',url=f'http://127.0.0.1:{port}/mcp'))
        self.assertIn('real HTTP',runtime.call('mcp:remote:echo',{'text':'real HTTP'}))

    def test_openai_bundle_owns_one_pool_shared_by_forks_and_workers(self):
        from dataclasses import replace
        import shutil
        from orchestration.config import Settings
        from openai_agent.runtime import build_bundle
        with tempfile.TemporaryDirectory() as temp:
            prompts=Path(temp)/'prompts'
            shutil.copytree(ROOT/'prompts',prompts)
            settings=replace(Settings.from_env(require_model=False), model='test', prompt_dir=prompts,
                workspace=Path(temp),agent_count=1,
                mcp_servers=(McpServerConfig('demo',True,'stdio',sys.executable,(str(SERVER),)),))
            bundle=build_bundle(settings)
            self.addCleanup(bundle.close)
            runtime=bundle.runtime
            pool=runtime.skill_base.mcp_runtime
            self.assertIs(runtime.pool.get('agent1').tool_dispatcher,runtime.tool_dispatcher)
            self.assertIs(runtime.fork_context('other').tool_dispatcher,runtime.tool_dispatcher)
            self.assertIn('mcp:demo:echo',runtime.messages[0]['content'])
            self.assertIn('bundle result',runtime._execute_work_command('mcp:demo:echo {"text":"bundle result"}'))
            bundle.close()
            self.assertFalse(pool._thread.is_alive())
