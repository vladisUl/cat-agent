"""Stage 7: native LiteRT and existing OpenAI share one authoritative TASK SYSTEM."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import io
from contextlib import redirect_stderr
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from orchestration.tasks import TaskStoreError
from agent_core_tools.priority_scheduler import PriorityScheduler
from agent_core_tools.task_worker import NativeTaskWorker
from orchestration_tools.contracts import ManagerTurn
from task_system.client import RemoteSystemRuntime
from task_system.engine import TaskEngine
from test_agent_core_tools_priority import FakeManager

ROOT = Path(__file__).resolve().parents[1]


class LocalRPC:
    def __init__(self, engine, peer):
        self.engine, self.peer = engine, peer

    def call(self, target, operation, *args, **kwargs):
        response = self.engine.call(target, operation, args, kwargs, peer=self.peer)
        return json.loads(json.dumps(
            response,
            default=lambda item: asdict(item) if is_dataclass(item) else item,
        ))


class SharedSystemTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        self.engine = TaskEngine(
            directory / "one.sqlite3",
            task_file=directory / "tasks-legacy.txt",
            event_file=directory / "events-legacy.json",
            alive=lambda peer: peer in {"native", "openai"},
        )
        self.addCleanup(self.engine.close)

        self.native_rpc = LocalRPC(self.engine, "native")
        self.cloud_rpc = LocalRPC(self.engine, "openai")
        self.cloud = RemoteSystemRuntime("openai", rpc=self.cloud_rpc)
        self.cloud.heartbeat(True)
        self.addCleanup(self.cloud.stop_worker)
        self.manager = FakeManager()
        self.manager.run_agent = lambda agent_id, task: self.manager.pool.run(task, agent_id)
        self.manager.process_task_query = lambda task_id, text, failed=False: ManagerTurn(
            "reply", f"QUERY {task_id}: {text}",
        )
        self.scheduler = PriorityScheduler()
        self.addCleanup(self.scheduler.close)
        self.native = NativeTaskWorker(
            self.scheduler, self.manager, rpc=self.native_rpc,
            agent_permissions={"agent_1": ("shell",)}, poll_seconds=3600,
        )
        self.native.start()
        self.addCleanup(self.native.close)

    def create(self, executor, method="task", text="SHARED TEST ACTION"):
        task = self.cloud_rpc.call(
            "system", "create_periodic_task",
            "shared-" + executor, text, ["shell"], 3600,
            method=method, executor=executor,
        )
        run_id = self.cloud_rpc.call("system", "run_task", task["task_id"])
        return task, run_id

    def await_started_completion(self):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.native.snapshot()["unacknowledged"]:
                return
            time.sleep(.005)
        self.fail("native task did not finish its model turn")

    def test_both_cores_ready_and_auto_prefers_openai_explicit_litert_runs_native(self):
        cores = {c["executor"]: c["state"] for c in self.cloud_rpc.call("system", "cores")}
        self.assertEqual(cores, {"openai": "READY", "litert": "READY"})
        task_auto, auto_id = self.create("auto", text="CLOUD ONLY")
        task_native, native_id = self.create("litert", text="NATIVE ONLY")
        # Native pull must skip the auto job while OpenAI is READY.
        self.assertTrue(self.native.poll_once())
        self.await_started_completion()
        self.assertFalse(self.native.poll_once())  # finish without replay
        self.assertEqual(self.engine.runs[native_id]["state"], "done")
        self.assertEqual(self.engine.runs[auto_id]["state"], "pending")
        self.assertEqual(self.manager.events, ["NATIVE ONLY"])
        cloud_offers = self.cloud.poll_due()
        self.assertEqual(len(cloud_offers), 1)
        event, = cloud_offers
        self.assertEqual(event.task_id, task_auto["task_id"])
        self.assertTrue(self.cloud.begin_run(event))
        self.cloud.finish_run(event, "done")
        self.assertTrue(self.cloud.flush_completions())
        self.assertEqual(self.engine.runs[auto_id]["state"], "done")
        self.assertEqual(self.engine.runs[auto_id]["executor"], "openai")
        self.assertEqual(self.engine.runs[native_id]["executor"], "litert")
        self.assertEqual(self.manager.events, ["NATIVE ONLY"])

    def test_auto_falls_to_native_when_openai_not_ready(self):
        self.cloud.stop_worker()
        states = {c["executor"]: c["state"] for c in self.cloud_rpc.call("system", "cores")}
        self.assertEqual(states["openai"], "NOT_READY")
        self.assertEqual(states["litert"], "READY")
        _, run_id = self.create("auto", text="FALLBACK NATIVE")
        self.assertTrue(self.native.poll_once())
        self.await_started_completion()
        self.assertFalse(self.native.poll_once())
        self.assertEqual(self.engine.runs[run_id]["executor"], "litert")
        self.assertEqual(self.engine.runs[run_id]["state"], "done")
        self.assertEqual(self.manager.events, ["FALLBACK NATIVE"])

    def test_query_from_native_is_read_and_acked_via_same_sqlite_as_openai(self):
        _, run_id = self.create("litert", method="query", text="NATIVE QUERY")
        self.assertTrue(self.native.poll_once())
        self.await_started_completion()
        self.assertFalse(self.native.poll_once())
        self.assertEqual(self.engine.runs[run_id]["state"], "done")
        inbox = self.cloud_rpc.call("notifications", "list")
        self.assertEqual(len(inbox), 1)
        self.assertEqual(inbox[0]["run_id"], run_id)
        self.assertIn("NATIVE QUERY", inbox[0]["text"])
        self.assertTrue(self.cloud_rpc.call("notifications", "ack", run_id))
        self.assertEqual(self.native_rpc.call("notifications", "list"), [])


class NativeTaskSystemHandshakeTests(unittest.TestCase):
    def test_missing_service_fails_before_native_model_initialization(self):
        from agent_core_tools.__main__ import main
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            with patch("task_system.client.TaskRPC.call",
                       side_effect=TaskStoreError("shared TASK SYSTEM offline")):
                with patch("orchestration_tools.litert_runtime.build_native_manager") as builder:
                    errors = io.StringIO()
                    with redirect_stderr(errors):
                        code = main([
                            "--model", str(root / "model.litertlm"),
                            "--system-file", str(root / "manager.md"),
                            "--agent-config", str(root / "agents.json"),
                            "--task-socket", str(root / "nonexistent.sock"),
                            "--enable-task-worker",
                        ])
                    self.assertEqual(code, 2)
                    self.assertIn("Shared TASK SYSTEM unavailable", errors.getvalue())
                    builder.assert_not_called()

    def test_wrong_service_ping_is_rejected(self):
        from agent_core_tools.__main__ import main
        with patch("task_system.client.TaskRPC.call", return_value="OTHER_SERVICE"):
            with patch("orchestration_tools.litert_runtime.build_native_manager") as builder:
                errors = io.StringIO()
                with redirect_stderr(errors):
                    code = main([
                        "--model", "/tmp/model.litertlm",
                        "--system-file", "/tmp/manager.md",
                        "--agent-config", "/tmp/agents.json",
                        "--enable-task-worker",
                    ])
                self.assertEqual(code, 2)
                self.assertIn("unexpected TASK SYSTEM handshake", errors.getvalue())
                builder.assert_not_called()


class NativeSharedLauncherTests(unittest.TestCase):
    def test_litert_systemd_service_runs_native_core(self):
        service = (ROOT / "systemd/cat-agent-litert.service").read_text(encoding="utf-8")
        self.assertIn("Type=simple", service)
        self.assertIn("ExecStart=/opt/cat-agent/start_native_core.sh", service)
        self.assertNotIn("Type=notify", service)
        self.assertNotIn("start_litert_agent.sh", service)


    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.model = self.root / "model.litertlm"
        self.model.write_bytes(b"test-only")
        self.prompt = self.root / "manager.md"
        self.prompt.write_text("Native manager", encoding="utf-8")
        self.agents = self.root / "agents.json"
        self.agents.write_text('{"agents": []}', encoding="utf-8")
        self.stub_config = self.root / "config-interpreter"
        self.stub_config.write_text(
            '#!/bin/sh\n'
            'printf "export LITERT_AGENT_MODEL_PATH=%s\\n" "$NATIVE_FAKE_PROFILE_MODEL"\n'
            'printf "export LITERT_AGENT_BACKEND=%s\\n" "$NATIVE_FAKE_BACKEND"\n'
            'printf "export LITERT_AGENT_CPU_THREADS=8\\n"\n'
            'if [ "$NATIVE_FAKE_BACKEND" = gpu ]; then\n'
            '  printf "export LITERT_AGENT_ACTIVATION_DATA_TYPE=fp32\\n"\n'
            '  printf "export LITERT_AGENT_VISION_BACKEND=none\\n"\n'
            '  printf "export LITERT_AGENT_SPECULATIVE=1\\n"\n'
            'fi\n'
            'if [ -n "$NATIVE_FAKE_VULKAN_DIR" ]; then\n'
            '  printf "export LITERT_AGENT_VULKAN_ICD_DIR=%s\\n" "$NATIVE_FAKE_VULKAN_DIR"\n'
            'fi\n'
            'if [ -n "$NATIVE_FAKE_D3D12_ADAPTER" ]; then\n'
            '  printf "export LITERT_AGENT_D3D12_ADAPTER=%s\\n" "$NATIVE_FAKE_D3D12_ADAPTER"\n'
            'fi\n'
            'printf "export CAT_AGENT_WORKSPACE=/opt/model\\n"\n',
            encoding="utf-8",
        )
        self.stub_config.chmod(0o755)
        self.stub_runtime = self.root / "fake-native-interpreter"
        self.stub_runtime.write_text(
            '#!/usr/bin/env python3\n'
            'import json, os, sys\n'
            'with open(os.environ["NATIVE_CAPTURE"], "w", encoding="utf-8") as f:\n'
            '    json.dump(sys.argv[1:], f)\n'
            'with open(os.environ["NATIVE_ENV_CAPTURE"], "w", encoding="utf-8") as f:\n'
            '    json.dump({key: os.getenv(key) for key in ("VK_DRIVER_FILES", "MESA_D3D12_DEFAULT_ADAPTER_NAME")}, f)\n',
            encoding="utf-8",
        )
        self.stub_runtime.chmod(0o755)
        self.capture = self.root / "args.json"
        self.capture_env = self.root / "environ.json"

    def launch(self, **changes):
        # Profile values may be exported in the caller's WSL shell. The
        # launcher is tested with a stub config exporter, not the caller's
        # selected LiteRT profile; inherited LITERT_AGENT_* must not leak in.
        env = {
            **{key: value for key, value in os.environ.items()
               if not key.startswith("LITERT_AGENT_")},
            "CAT_AGENT_NATIVE_CONFIG_PYTHON": str(self.stub_config),
            "CAT_AGENT_TOOLS_PYTHON": str(self.stub_runtime),
            "CAT_AGENT_NATIVE_MODEL_PATH": str(self.model),
            "CAT_AGENT_NATIVE_MANAGER_PROMPT": str(self.prompt),
            "CAT_AGENT_NATIVE_AGENT_CONFIG": str(self.agents),
            "NATIVE_FAKE_PROFILE_MODEL": str(self.model),
            "NATIVE_FAKE_BACKEND": "cpu",
            "NATIVE_CAPTURE": str(self.capture),
            "NATIVE_ENV_CAPTURE": str(self.capture_env),
        }
        # Ensure host-level admin overrides cannot change default tests.
        env.pop("CAT_AGENT_TASK_SOCKET", None)
        env.pop("CAT_AGENT_NATIVE_CORE_SOCKET", None)
        for key, value in changes.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = str(value)
        return subprocess.run(
            ["bash", str(ROOT / "start_native_core.sh")],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
        )

    def test_shared_default_launch_uses_one_task_socket_and_separate_native_core_socket(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.capture.read_text(encoding="utf-8"))
        self.assertEqual(args[:2], ["-m", "agent_core_tools"])
        opts = args[2:]
        self.assertEqual(opts[opts.index("--task-socket") + 1], "/run/cat-agent/tasks.sock")
        self.assertEqual(opts[opts.index("--socket") + 1], "/run/cat-agent/litert-tools.sock")
        self.assertEqual(opts[opts.index("--model") + 1], str(self.model))
        self.assertEqual(opts[opts.index("--cpu-threads") + 1], "8")
        self.assertIn("--enable-task-worker", opts)
        self.assertNotIn("--disable-mqtt", opts)
        self.assertEqual(opts[opts.index("--backend") + 1], "cpu")
        self.assertEqual(opts[opts.index("--vision-backend") + 1], "cpu")
        self.assertNotIn("--speculative", opts)
        self.assertIn("--enable-shell", opts)

    def test_inherited_gpu_profile_cannot_contaminate_cpu_launch(self):
        with patch.dict(os.environ, {
            "LITERT_AGENT_BACKEND": "gpu",
            "LITERT_AGENT_VISION_BACKEND": "none",
            "LITERT_AGENT_ACTIVATION_DATA_TYPE": "fp32",
            "LITERT_AGENT_SPECULATIVE": "1",
        }):
            result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        opts = json.loads(self.capture.read_text(encoding="utf-8"))[2:]
        self.assertEqual(opts[opts.index("--backend") + 1], "cpu")
        self.assertEqual(opts[opts.index("--vision-backend") + 1], "cpu")
        self.assertNotIn("--activation-dtype", opts)
        self.assertNotIn("--speculative", opts)

    def test_configured_task_socket_and_explicit_model_are_preserved(self):
        alternate_model = self.root / "chosen.litertlm"
        alternate_model.write_bytes(b"fake")
        result = self.launch(
            CAT_AGENT_NATIVE_MODEL_PATH=alternate_model,
            CAT_AGENT_TASK_SOCKET="/tmp/shared-test.sock",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(self.capture.read_text(encoding="utf-8"))[2:]
        self.assertEqual(args[args.index("--model") + 1], str(alternate_model))
        self.assertEqual(args[args.index("--task-socket") + 1], "/tmp/shared-test.sock")

    def test_gpu_profile_selects_gpu_and_passes_profile_options(self):
        result = self.launch(NATIVE_FAKE_BACKEND="gpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        opts = json.loads(self.capture.read_text(encoding="utf-8"))[2:]
        self.assertEqual(opts[opts.index("--backend") + 1], "gpu")
        self.assertEqual(opts[opts.index("--vision-backend") + 1], "none")
        self.assertEqual(opts[opts.index("--activation-dtype") + 1], "fp32")
        self.assertIn("--speculative", opts)
        self.assertNotIn("--cpu-threads", opts)
        self.assertNotIn("--disable-mqtt", opts)

    def test_gpu_profile_resolves_only_icd_and_selects_nvidia(self):
        icd_dir = self.root / "dozen" / "icd.d"
        icd_dir.mkdir(parents=True)
        icd = icd_dir / "dzn_icd.x86_64.json"
        icd.write_text("{}", encoding="utf-8")
        result = self.launch(
            NATIVE_FAKE_BACKEND="gpu",
            NATIVE_FAKE_VULKAN_DIR=icd_dir,
            NATIVE_FAKE_D3D12_ADAPTER="NVIDIA",
            VK_DRIVER_FILES="/tmp/inherited-lvp.json",
            MESA_D3D12_DEFAULT_ADAPTER_NAME="Intel",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        native_env = json.loads(self.capture_env.read_text(encoding="utf-8"))
        self.assertEqual(native_env["VK_DRIVER_FILES"], str(icd))
        self.assertEqual(native_env["MESA_D3D12_DEFAULT_ADAPTER_NAME"], "NVIDIA")
        self.assertIn("Native CORE Vulkan ICD:", result.stderr)

    def test_missing_vulkan_icd_dir_fails_before_engine(self):
        result = self.launch(
            NATIVE_FAKE_BACKEND="gpu",
            NATIVE_FAKE_VULKAN_DIR=self.root / "missing-driver",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("Vulkan ICD directory missing", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_ambiguous_vulkan_icd_dir_fails_before_engine(self):
        icd_dir = self.root / "icd.d"
        icd_dir.mkdir()
        (icd_dir / "a.json").write_text("{}", encoding="utf-8")
        (icd_dir / "b.json").write_text("{}", encoding="utf-8")
        result = self.launch(
            NATIVE_FAKE_BACKEND="gpu",
            NATIVE_FAKE_VULKAN_DIR=icd_dir,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("expected exactly one Vulkan ICD JSON", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_cpu_profile_clears_inherited_vulkan_override(self):
        result = self.launch(
            VK_DRIVER_FILES="/tmp/inherited-lvp.json",
            MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        native_env = json.loads(self.capture_env.read_text(encoding="utf-8"))
        self.assertIsNone(native_env["VK_DRIVER_FILES"])
        self.assertIsNone(native_env["MESA_D3D12_DEFAULT_ADAPTER_NAME"])

    def test_unknown_backend_is_rejected_before_engine(self):
        result = self.launch(NATIVE_FAKE_BACKEND="unconfigured")
        self.assertEqual(result.returncode, 2)
        self.assertIn("Unsupported Native LiteRT backend", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_rejects_socket_collision_with_openai(self):
        result = self.launch(CAT_AGENT_NATIVE_CORE_SOCKET="/run/cat-agent/openai.sock")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be distinct", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_missing_model_fails_before_starting_runtime(self):
        result = self.launch(CAT_AGENT_NATIVE_MODEL_PATH=self.root / "absent.litertlm")
        self.assertEqual(result.returncode, 2)
        self.assertIn("model file missing", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
