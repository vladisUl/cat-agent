import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from orchestration.agent import AgentWorker
from orchestration.assistant_manager import AssistantManagerRuntime
from orchestration.manager import AutonomousTaskExecution
from orchestration.model_client import ChatResponse
from orchestration.pool import AgentPool
from orchestration.prompt_store import PromptStore
from orchestration.protocol import parse_manager_output
from orchestration.skills import Skill, SkillBase, SkillBaseError
from orchestration.system_events import SystemRuntime, SystemEvent
from orchestration.tasks import TaskStore
from orchestration.tool_catalog import ToolCatalog
from orchestration.tool_dispatcher import ToolDispatcher
from orchestration.workspace_command_runtime import CommandRuntime

ROOT=Path(__file__).resolve().parents[1]
NAME='mcp:test:echo'


class FakeMcp:
    def __init__(self):
        self.calls=[]
        self.response='MCP_RESULT\n{"content":[{"type":"text","text":"real answer"}],"isError":false}'
    def skills(self):
        return (Skill(NAME,'Echo','Call /work#'+NAME+' {JSON}; schema: {"type":"object"}'),)
    def call(self, name, arguments):
        self.calls.append((name,arguments))
        return self.response


class FakeModel:
    def __init__(self, replies):
        self.replies=list(replies)
        self.calls=[]
    def chat(self,messages):
        self.calls.append([dict(x) for x in messages])
        return ChatResponse(self.replies.pop(0),None,None,.001)
    def reset_to_base(self,messages):
        pass
    def fork(self,label):
        return FakeModel([])


class McpDispatchTest(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.prompts=self.root/'prompts'
        shutil.copytree(ROOT/'prompts',self.prompts)
        self.mcp=FakeMcp()
        self.catalog=ToolCatalog(SkillBase(self.prompts/'prompt_base.txt'),self.mcp)
        self.dispatcher=ToolDispatcher(self.mcp)
        self.commands=CommandRuntime(self.root,('shell','mqtt','read_pic',NAME),max_file_bytes=4096,timeout_seconds=2)

    def runtime(self, manager_replies, agent_replies=()):
        manager=FakeModel(manager_replies)
        agent=FakeModel(agent_replies)
        prompts=PromptStore(self.prompts,1)
        worker=AgentWorker('agent1',agent,prompts,self.root,max_steps=5,max_file_bytes=4096,
                           command_timeout_seconds=2,tool_dispatcher=self.dispatcher)
        runtime=AssistantManagerRuntime(manager,self.catalog,prompts,AgentPool([worker]),
            SystemRuntime(TaskStore(self.root/'tasks.txt')),max_steps=6,tool_dispatcher=self.dispatcher)
        return runtime,manager,agent,worker

    def test_json_is_not_shell_and_manager_receives_actual_result(self):
        text="O'Brien $(touch SHOULD_NOT_EXIST); \\\"nested\\\""
        command='mcp:test:echo '+json.dumps({'text':text,'nested':{'a':[1,True]}})
        runtime,model,_,_=self.runtime(['/work#'+command,'REPLY\nreal answer'])
        with patch.object(CommandRuntime,'execute',side_effect=AssertionError('shell reached')):
            self.assertEqual(runtime.user_message('echo').text,'real answer')
        self.assertEqual(self.mcp.calls,[(NAME,{'text':text,'nested':{'a':[1,True]}})])
        self.assertIn('real answer',model.calls[1][-1]['content'])
        self.assertFalse((self.root/'SHOULD_NOT_EXIST').exists())
        self.assertIn(NAME,runtime._base_messages[0]['content'])

    def test_agent_uses_same_dispatcher(self):
        _,_,model,worker=self.runtime([],['/work#mcp:test:echo {"text":"x"}','{"result":"real answer"}'])
        with patch.object(CommandRuntime,'execute',side_effect=AssertionError('shell reached')):
            outcome=worker.start('echo',self.catalog.require((NAME,)))
        self.assertEqual(outcome.text,'real answer')
        self.assertIn('real answer',model.calls[1][-1]['content'])

    def test_saved_task_survives_reload_and_resolves_mcp(self):
        runtime,_,_,worker=self.runtime([],['/work#mcp:test:echo {"text":"x"}','{"done":true}'])
        result=runtime._execute_work_command('task_timer.sh 60 mcp:test:echo -- "echo periodically"')
        tasks=TaskStore(self.root/'tasks.txt')
        task=tasks.list()[0]
        self.assertEqual(task.skills,(NAME,))
        self.assertEqual(task.executor,'auto')
        runtime.system_runtime=SystemRuntime(tasks)
        execution=runtime.begin_autonomous_task(SystemEvent('timer','test','',0,task_id=task.task_id))
        self.assertIsInstance(execution,AutonomousTaskExecution)
        self.assertIsNone(runtime.step_autonomous_task(execution))
        self.assertIsNotNone(runtime.step_autonomous_task(execution))
        self.assertEqual(self.mcp.calls,[(NAME,{'text':'x'})])

    def test_one_shot_task_resolves_mcp(self):
        runtime,_,_,_=self.runtime([],['/work#mcp:test:echo {"text":"x"}','{"result":"real answer"}'])
        steps=runtime._execute_work_steps('task_timer.sh 0 mcp:test:echo -- "echo once"')
        self.assertEqual(runtime._finish_steps(steps),'real answer')

    def test_namespace_in_legacy_task_protocol(self):
        directive=parse_manager_output('DELEGATE mcp:test:echo\necho text')
        self.assertFalse(directive.error)
        self.assertEqual(directive.skills,(NAME,))

    def test_catalog_conflict_is_explicit(self):
        class Collision(FakeMcp):
            def skills(self): return (Skill('shell','conflict','conflict'),)
        with self.assertRaises(SkillBaseError):
            ToolCatalog(SkillBase(self.prompts/'prompt_base.txt'),Collision())

    def test_malformed_unknown_and_unassigned_never_fall_through(self):
        for command in ('mcp:test:echo','mcp:test:echo []','mcp:test:echo {"x":NaN}',
                        'mcp:test:echo {"x":1,"x":2}','mcp:bad:tool {}','mcp:test:echo {} ; touch file'):
            with self.subTest(command=command):
                self.assertIn('SYSTEM_ERROR',self.dispatcher.dispatch(command,self.commands))
        restricted=CommandRuntime(self.root,('shell',),max_file_bytes=4096,timeout_seconds=2)
        self.assertIn('not assigned',self.dispatcher.dispatch('mcp:test:echo {}',restricted))
        self.assertEqual(self.mcp.calls,[])

    def test_uncertain_call_is_not_replayed_even_with_different_json_spacing(self):
        self.mcp.response='SYSTEM_ERROR\nMCP outcome_unknown: connection lost'
        self.dispatcher.dispatch('mcp:test:echo {"text":"x"}',self.commands)
        self.assertIn('replay refused',self.dispatcher.dispatch('mcp:test:echo { "text": "x" }',self.commands))
        self.assertEqual(len(self.mcp.calls),1)

    def test_legacy_commands_not_consumed_by_dispatcher(self):
        for command in ('printf ok','mqtt_sub.sh topic field','mqtt_pub.sh topic state=ON',
                        'read_pic.sh image.png','timer.sh list','task_timer.sh 60 shell -- "test"'):
            with self.subTest(command=command):
                self.assertIsNone(self.dispatcher.dispatch(command,self.commands))
        runtime,_,_,_=self.runtime([])
        self.assertIn('shell-ok',runtime._execute_work_command('printf shell-ok'))
        with patch('orchestration.assistant_manager.read_picture',return_value=[{'type':'image_url','image_url':{'url':'test'}}]):
            self.assertIsInstance(runtime._execute_work_command('read_pic.sh image.png'),list)
        self.assertEqual(self.mcp.calls,[])

    def test_fork_shares_catalog_dispatcher_and_base(self):
        runtime,_,_,_=self.runtime([])
        fork=runtime.fork_context('test')
        self.assertIs(fork.skill_base,runtime.skill_base)
        self.assertIs(fork.tool_dispatcher,runtime.tool_dispatcher)
        self.assertEqual(fork.messages,runtime._base_messages)
        self.mcp.response='SYSTEM_ERROR\nMCP unavailable'
        before=runtime._base_messages[0]['content']
        runtime._execute_work_command('mcp:test:echo {}')
        runtime.reset_for_new_session()
        self.assertEqual(runtime.messages[0]['content'],before)
        self.assertEqual(runtime.skill_base.require((NAME,))[0].name,NAME)
