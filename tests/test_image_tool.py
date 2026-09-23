from __future__ import annotations

import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from orchestration.image_tool import read_picture
from openai_agent.model_client import OpenAICompatibleChatClient
import test_assistant_manager as fixtures

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/a9sAAAAASUVORK5CYII=")


class ImageToolTest(unittest.TestCase):
    def test_bytes_are_embedded_and_immutable_after_file_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            path = data / "my cat.png"
            path.write_bytes(PNG)
            runtime = SimpleNamespace(cwd=root, root=root, skill_names={"read_pic"})
            result = read_picture('read_pic.sh "/my cat.png"', runtime, SimpleNamespace(supports_images=True))
            path.write_bytes(b"changed")
            url = result[1]["image_url"]["url"]
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), PNG)

    def test_errors_and_unsupported_backend(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as outside:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            (data / "cat.png").write_bytes(PNG)
            (Path(outside) / "cat.png").write_bytes(PNG)
            (data / "link.png").symlink_to(Path(outside) / "cat.png")
            runtime = SimpleNamespace(cwd=root, root=root, skill_names={"read_pic"})
            client = SimpleNamespace(supports_images=True)
            for command in ('read_pic.sh', 'read_pic.sh /missing.png', 'read_pic.sh /link.png', 'read_pic.sh /cat.png ; echo bad', 'read_pic.sh ../cat.png'):
                with self.subTest(command=command):
                    self.assertIn("SYSTEM_ERROR", read_picture(command, runtime, client))
            self.assertIsInstance(read_picture("read_pic.sh cat.png", runtime, client), list)
            self.assertIsInstance(
                read_picture(
                    "read_pic.sh",
                    runtime,
                    client,
                    input_text="cat.png",
                ),
                list,
            )
            self.assertIsInstance(
                read_picture(
                    "read_pic.sh cat.png",
                    runtime,
                    client,
                    input_text="ignored.png",
                ),
                list,
            )
            self.assertIn("supported only", read_picture("read_pic.sh /cat.png", runtime, SimpleNamespace()))
            with patch.dict("os.environ", {"CAT_AGENT_MAX_IMAGE_BYTES": "4"}):
                self.assertIn("exceeds", read_picture("read_pic.sh /cat.png", runtime, client))
            with patch.dict("os.environ", {"CAT_AGENT_ENABLED_SKILLS": "shell,mqtt"}):
                self.assertIn("disabled", read_picture("read_pic.sh cat.png", runtime, client))
            runtime.skill_names = {"mqtt"}
            self.assertIn("not assigned", read_picture("read_pic.sh cat.png", runtime, client))
            self.assertIsNone(read_picture("echo hello", runtime, client))

    def test_manager_reads_image_and_retains_it_for_followup(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = fixtures.AssistantManagerTest()._runtime(Path(temp), [
                "REPLY\nЧат создан", "/work#read_pic.sh /cat.png", "REPLY\nКот", "REPLY\nЦветы",
            ])
            client.supports_images = True
            (runtime._direct_runtime.root / "data").mkdir()
            (runtime._direct_runtime.root / "data" / "cat.png").write_bytes(PNG)
            runtime.user_message("Чат")
            self.assertEqual(runtime.user_message("Что на /cat.png?").text, "Кот")
            self.assertEqual(runtime.user_message("А рядом?").text, "Цветы")
            images = [m for m in client.calls[-1] if isinstance(m["content"], list)]
            self.assertEqual(len(images), 1)
            self.assertEqual(images[0]["content"][1]["type"], "image_url")

    def test_worker_receives_image_in_its_own_context(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime, client = fixtures.AssistantManagerTest()._runtime(Path(temp), [
                "/work#read_pic.sh /cat.png", '{"result":"Кот"}',
            ])
            client.supports_images = True
            (runtime._direct_runtime.root / "data").mkdir(exist_ok=True)
            (runtime._direct_runtime.root / "data" / "cat.png").write_bytes(PNG)
            worker = runtime.pool.acquire()
            worker.begin("Что на /cat.png?", runtime.skill_base.require(("read_pic",)), method="query")
            self.assertIsNone(worker.step())
            outcome = worker.step()
            self.assertEqual(outcome.text, "Кот")
            self.assertEqual(client.calls[-1][-1]["content"][1]["type"], "image_url")

    def test_openai_payload_keeps_image_and_followup_history(self):
        client = OpenAICompatibleChatClient(
            api_base_url="http://unused/v1", model="vision", timeout_seconds=10,
            retries=0, retry_delay_seconds=0, max_output_tokens=100,
            temperature=0, top_p=1, reasoning_effort="none", label="manager", api_key="",
        )
        image = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}]}
        history = [{"role": "system", "content": "base"}, image, {"role": "assistant", "content": "Кот"}, {"role": "user", "content": "А рядом?"}]
        payload = json.loads(json.dumps(client._payload(history, stream=True)))
        self.assertEqual(payload["messages"], history)


def load_litert_adapter():
    # Exercise adapter control flow without importing a native LiteRT library.
    path = Path(__file__).resolve().parents[1] / "src/litert_agent/model_client.py"
    spec = importlib.util.spec_from_file_location("vision_adapter_under_test", path)
    module = importlib.util.module_from_spec(spec)
    stub = SimpleNamespace(SamplerConfig=lambda **kwargs: kwargs)
    with patch.dict("sys.modules", {"litert_lm": stub, spec.name: module}):
        spec.loader.exec_module(module)
    return module


class LiteRTImageContextTest(unittest.TestCase):
    def test_migration_followup_and_reset(self):
        module = load_litert_adapter()
        sent = []
        def stream(message):
            sent.append(deepcopy(message))
            yield {"content": [{"type": "text", "text": "REPLY\nКот"}]}
        conversation = SimpleNamespace(
            send_message_async=stream, close=Mock(),
            get_benchmark_info=lambda: SimpleNamespace(last_prefill_token_count=20, last_decode_token_count=4),
        )
        engine = SimpleNamespace(create_conversation=Mock(return_value=conversation))
        client = module.LiteRTChatClient(engine, max_output_tokens=100, temperature=0, top_p=1, reasoning_effort="none", label="manager")
        base = [{"role": "system", "content": "base"}]
        history = base + [{"role": "user", "content": "Что на cat.png?"}, {"role": "assistant", "content": "/work#read_pic.sh cat.png"}]
        image = {"role": "user", "content": [{"type": "text", "text": "image"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(PNG).decode()}}]}
        messages = history + [image]
        answer = client.chat(messages)
        self.assertEqual(engine.create_conversation.call_args.kwargs["messages"], history)
        self.assertEqual(sent[0]["content"][1], {"type": "image", "blob": base64.b64encode(PNG).decode()})
        messages += [{"role": "assistant", "content": answer.content}, {"role": "user", "content": "А рядом?"}]
        client.chat(messages)
        engine.create_conversation.assert_called_once()
        self.assertEqual(sent[-1], messages[-1])
        client._base_messages = base
        client._session = SimpleNamespace(_ptr=1, close=Mock())
        client._lib = SimpleNamespace(litert_lm_session_rewind_to_checkpoint=Mock(return_value=0))
        client._checkpoint_ready = True
        client.reset_to_base(base)
        conversation.close.assert_called_once()
        self.assertIsNone(client._vision_conversation)
        self.assertEqual(client._synced_messages, base)
        client.close()

    def test_failed_image_turn_does_not_resume_stale_text_kv(self):
        module = load_litert_adapter()
        conversation = SimpleNamespace(send_message_async=Mock(side_effect=RuntimeError("invalid image")), close=Mock())
        client = module.LiteRTChatClient(SimpleNamespace(create_conversation=lambda **kw: conversation), max_output_tokens=10, temperature=0, top_p=1, reasoning_effort="none", label="manager")
        with self.assertRaisesRegex(module.ModelClientError, "invalid image"):
            client.chat([{"role": "system", "content": "base"}, {"role": "user", "content": [{"type": "text", "text": "image"}]}])
        self.assertIsNone(client._vision_conversation)
        self.assertEqual(client._synced_messages, [])
        conversation.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
