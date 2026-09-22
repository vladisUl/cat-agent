import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import yaml

from orchestration.config import Settings
from orchestration.mcp_config import parse_mcp_config, load_mcp_config
from orchestration.skills import SkillBase
from orchestration.tool_catalog import build_tool_catalog
from orchestration.yaml_config import load_app_config

ROOT = Path(__file__).resolve().parents[1]


class McpConfigTest(unittest.TestCase):
    def setUp(self):
        self.entry = dict(name="local", enabled=True, transport="stdio", command="python", args=["server.py"])

    def test_both_transports_and_environment_references(self):
        entries = [self.entry, dict(name="remote", enabled=True, transport="streamable_http",
                    url="https://example.org/mcp", headers_env={"Authorization": "MCP_AUTH"})]
        config = parse_mcp_config({"servers": entries})
        self.assertEqual(config[0].args, ("server.py",))
        self.assertEqual(config[1].headers_env, (("Authorization", "MCP_AUTH"),))
        self.assertEqual(config[0].call_timeout_seconds, 30)
        self.assertTrue(config[0].manager)
        self.assertEqual(config[0].description, "")
        explicit = parse_mcp_config({"servers":[dict(
            self.entry, manager=False, description="Excel work"
        )]})
        self.assertFalse(explicit[0].manager)
        self.assertEqual(explicit[0].description, "Excel work")

    def test_invalid_config(self):
        for updates in ({"name":"a:b"}, {"enabled":"false"}, {"transport":"sse"},
                        {"args":"server.py"}, {"command":""}, {"call_timeout_seconds":0},
                        {"connect_timeout_seconds":float("nan")}, {"manager":"false"}, {"manager":False},
                        {"description":123}, {"description":"bad" + "\n" + "line"}, {"unknown":1}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                parse_mcp_config({"servers":[dict(self.entry, **updates)]})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_mcp_config({"servers":[self.entry, self.entry]})
        for url in ("file:///tmp/test", "http://user:secret@host/mcp", "https://"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                parse_mcp_config({"servers":[dict(name="http", enabled=True, transport="streamable_http", url=url)]})

    def test_yaml_and_settings_use_same_config(self):
        data = yaml.safe_load((ROOT / "cat-agent.yaml").read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            data["mcp"] = {"servers":[self.entry]}
            path.write_text(yaml.safe_dump(data))
            with patch.dict(os.environ, {"CAT_AGENT_CONFIG": str(path)}):
                expected = load_app_config(path).mcp
                self.assertEqual(load_mcp_config(), expected)
                self.assertEqual(Settings.from_env(require_model=False).mcp_servers, expected)
            del data["mcp"]
            path.write_text(yaml.safe_dump(data))
            self.assertEqual(load_app_config(path).mcp, ())

    def test_disabled_has_no_runtime_or_sdk_import(self):
        config = parse_mcp_config({"servers":[dict(self.entry, enabled=False)]})
        with patch('orchestration.mcp_runtime.McpRuntime', side_effect=AssertionError('must not start')):
            catalog = build_tool_catalog(ROOT / 'prompts/prompt_base.txt', config)
        self.assertIsInstance(catalog, SkillBase)

    def test_no_mcp_dependency_for_ordinary_core(self):
        script = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'mcp' or name.startswith('mcp.') or name in {'anyio', 'jsonschema', 'httpx2'}:
        raise AssertionError('optional dependency imported: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from pathlib import Path
from orchestration.tool_catalog import build_tool_catalog
from orchestration.assistant_manager import AssistantManagerRuntime
from orchestration.agent import AgentWorker
from openai_agent.runtime import build_bundle
catalog = build_tool_catalog(
    Path('prompts/prompt_base.txt'),
    (),
    skills_dir=Path('skills'),
)
names = catalog.names()
assert names[:3] == ('shell', 'mqtt', 'read_pic')
assert 'prognoz' in names
assert not any(name.startswith('mcp:') for name in names)
'''
        result = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True,
                                env={**os.environ, 'PYTHONPATH':str(ROOT/'src')}, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
