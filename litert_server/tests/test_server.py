import concurrent.futures
from contextlib import contextmanager
from copy import deepcopy
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from litert_server.runtime import LiteRTRuntime, TextContext
from litert_server.server import InferenceService, RequestError, Server, load_config, parse_request


CONFIG = {"max_num_tokens": 8192, "temperature": 0.0, "top_p": 1.0}
BASE = {"role": "system", "content": "Ты помощник. " * 100}


def request(text="привет", history=None, **fields):
    return {"model": "test", "messages": deepcopy(history or [BASE]) + [
        {"role": "user", "content": text}
    ], **fields}


class FakeContext:
    def __init__(self, runtime, base, options):
        self.runtime, self.base, self.options = runtime, base, options
        self.base_tokens = 100
        self.resident = self.base_tokens
        self.history = deepcopy(base)
        self.closed = False
        self.calls = []

    def prefill(self, messages, continuation):
        self.calls.append((deepcopy(messages), continuation))
        if continuation:
            if messages[:-1] != self.history:
                raise AssertionError("Wrong native history reused")
            evaluated = 5
        else:
            self.resident = self.base_tokens
            self.history = deepcopy(self.base)
            evaluated = 5 + 7 * (len(messages) - len(self.base) - 1)
        cached = self.resident
        self.resident += evaluated
        self.history = deepcopy(messages)
        return cached, evaluated

    def decode(self):
        self.runtime.active += 1
        self.runtime.peak = max(self.runtime.peak, self.runtime.active)
        try:
            if self.runtime.fail:
                yield "partial"
                raise RuntimeError("native decode failed")
            yield "При"
            yield "вет!"
        finally:
            self.runtime.active -= 1
            self.runtime.drained += 1

    def finish(self):
        self.resident += 3
        self.history.append({"role": "assistant", "content": "Привет!"})
        return 3

    def close(self):
        if self.runtime.active:
            raise AssertionError("Native pointer freed while decode is active")
        self.closed = True


class FakeRuntime:
    def __init__(self):
        self.contexts = []
        self.fail = False
        self.active = self.peak = self.drained = 0
        self.closed = False

    def new_context(self, base, options):
        context = FakeContext(self, deepcopy(base), options)
        self.contexts.append(context)
        return context

    def close(self):
        self.closed = True


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = InferenceService(self.runtime, "test", CONFIG)
        self.addCleanup(self.service.close)

    def test_repeated_independent_requests_reuse_base_without_old_answer(self):
        first = self.service.complete(request())
        second = self.service.complete(request("другая задача"))
        self.assertEqual(len(self.runtime.contexts), 1)
        context = self.runtime.contexts[0]
        self.assertFalse(context.calls[1][1])
        self.assertEqual(context.history, request("другая задача")["messages"] + [
            {"role": "assistant", "content": "Привет!"}])
        self.assertEqual(first[1]["prompt_tokens_details"]["cached_tokens"], 0)
        self.assertEqual(first[1]["prompt_tokens"], 105)
        self.assertEqual(second[1]["prompt_tokens_details"]["cached_tokens"], 100)
        self.assertEqual(second[1]["prompt_tokens"], 105)

    def test_continuation_includes_only_new_turn_in_prefill(self):
        original = request()
        content, _, _ = self.service.complete(original)
        history = original["messages"] + [{"role": "assistant", "content": content}]
        _, usage, _ = self.service.complete(request("продолжай", history=history))
        self.assertTrue(self.runtime.contexts[0].calls[-1][1])
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 108)
        self.assertEqual(usage["prompt_tokens"] - 108, 5)

    def test_changed_assistant_history_is_replayed(self):
        self.service.complete(request())
        history = request()["messages"] + [{"role": "assistant", "content": "edited"}]
        self.service.complete(request("дальше", history=history))
        self.assertFalse(self.runtime.contexts[0].calls[-1][1])

    def test_interleaved_histories_rewind_safely(self):
        one, two = request("one"), request("two")
        self.service.complete(one)
        self.service.complete(two)
        history = one["messages"] + [{"role": "assistant", "content": "Привет!"}]
        self.service.complete(request("one continues", history=history))
        self.assertFalse(self.runtime.contexts[0].calls[-1][1])
        self.assertNotIn({"role": "user", "content": "two"}, self.runtime.contexts[0].history)

    def test_distinct_system_prompts_and_sampler_settings_have_separate_slots(self):
        self.service.complete(request())
        self.service.complete(request(history=[{"role": "system", "content": "other"}]))
        self.service.complete(request())
        self.assertEqual(len(self.runtime.contexts), 2)
        self.service.complete(request(temperature=0.7))
        self.assertTrue(self.runtime.contexts[1].closed)  # LRU, not most recently used
        self.assertFalse(self.runtime.contexts[0].closed)

    def test_failure_discards_partial_context(self):
        self.runtime.fail = True
        with self.assertRaisesRegex(RuntimeError, "native decode"):
            self.service.complete(request())
        self.assertFalse(self.service.slots)
        self.assertTrue(self.runtime.contexts[0].closed)
        self.runtime.fail = False
        _, usage, _ = self.service.complete(request())
        self.assertEqual(usage["prompt_tokens_details"]["cached_tokens"], 0)

    def test_disconnect_drains_native_decode_before_free(self):
        def emit(_):
            raise BrokenPipeError()
        with self.assertRaises(ConnectionError):
            self.service.complete(request(), emit)
        self.assertEqual(self.runtime.drained, 1)
        self.assertTrue(self.runtime.contexts[0].closed)
        self.assertFalse(self.service.slots)

    def test_truncated_reply_does_not_resume_incomplete_native_turn(self):
        original = request(max_tokens=3)
        content, _, reason = self.service.complete(original)
        self.assertEqual(reason, "length")
        history = original["messages"] + [{"role": "assistant", "content": content}]
        self.service.complete(request("continue", history=history, max_tokens=3))
        self.assertFalse(self.runtime.contexts[0].calls[-1][1])

    def test_native_inference_is_serialized(self):
        entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
        def emit(_):
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test timed out")
        def second():
            second_started.set()
            return self.service.complete(request("second"))
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first = pool.submit(self.service.complete, request(), emit)
            self.assertTrue(entered.wait(5))
            other = pool.submit(second)
            self.assertTrue(second_started.wait(5))
            release.set()
            first.result(5)
            other.result(5)
        self.assertEqual(self.runtime.peak, 1)


class ValidationTests(unittest.TestCase):
    def test_both_token_limit_names_and_large_cat_agent_limit(self):
        for key in ("max_tokens", "max_completion_tokens"):
            _, options = parse_request(request(**{key: 128000}), "test", CONFIG)
            self.assertEqual(options["max_output_tokens"], 8191)
        with self.assertRaises(RequestError):
            parse_request(request(max_tokens=1, max_completion_tokens=2), "test", CONFIG)

    def test_reject_invalid_and_unsupported_fields_before_native_work(self):
        invalid = [
            {"temperature": float("nan")}, {"top_p": 1.1}, {"seed": True},
            {"max_tokens": False}, {"max_tokens": 0}, {"stream": "false"},
            {"n": True}, {"n": 2}, {"tools": []}, {"stop": ["end"]},
            {"reasoning_effort": "high"}, {"thinking": True},
            {"stream_options": {"include_usage": "yes"}},
            {"model": "other"},
        ]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(RequestError):
                parse_request(request(**fields), "test", CONFIG)

    def test_text_blocks_normalize_but_multimodal_does_not_disappear(self):
        body = request()
        body["messages"][-1]["content"] = [{"type": "text", "text": "Hi"}]
        self.assertEqual(parse_request(body, "test", CONFIG)[0][-1]["content"], "Hi")
        body["messages"][-1]["content"].append({"type": "image_url", "image_url": {"url": "x"}})
        with self.assertRaises(RequestError):
            parse_request(body, "test", CONFIG)

    def test_config_current_user_shape_bom_relative_path_and_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "server-config.json"
            path.write_text(json.dumps({"default": {"temperature": 0.0}, "models": {
                "gemma4-12b": {"backend": "cpu", "cpu_thread_count": 6,
                    "speculative_decoding": True, "max_num_tokens": 8192,
                    "model_path": "model.litertlm"}}}), encoding="utf-8-sig")
            name, config = load_config(path)
            self.assertEqual(name, "gemma4-12b")
            self.assertEqual(config["temperature"], 0.0)
            self.assertEqual(Path(config["model_path"]), Path(temp) / "model.litertlm")

    def test_incompatible_native_abi_is_rejected_before_loading_library(self):
        with patch("importlib.metadata.version", return_value="1.0.0"):
            with self.assertRaisesRegex(RuntimeError, "checkpoint C ABI"):
                LiteRTRuntime("test", {})


@contextmanager
def running_server():
    runtime = FakeRuntime()
    service = InferenceService(runtime, "test", CONFIG)
    server = Server(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, service
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        service.close()


class HTTPTests(unittest.TestCase):
    def send(self, port, path, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)
        conn.request("POST" if body is not None else "GET", path,
                     json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, response.read().decode("utf-8")

    def test_json_sse_usage_health_and_error(self):
        with running_server() as (port, service):
            self.assertEqual(self.send(port, "/health")[0], 200)
            status, raw = self.send(port, "/v1/models")
            self.assertEqual(json.loads(raw)["data"][0]["id"], "test")
            status, raw = self.send(port, "/v1/chat/completions", request())
            self.assertEqual(status, 200)
            first = json.loads(raw)
            self.assertEqual(first["choices"][0]["message"]["content"], "Привет!")
            status, raw = self.send(port, "/v1/chat/completions", request(stream=True))
            self.assertEqual(status, 200)
            events = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
            self.assertEqual(events[-1], "[DONE]")
            chunks = [json.loads(event) for event in events[:-1]]
            text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
            self.assertEqual(text, "Привет!")
            self.assertEqual(chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 100)
            self.assertEqual(len({c["id"] for c in chunks}), 1)
            self.assertEqual(self.send(port, "/missing")[0], 404)
            status, raw = self.send(port, "/v1/chat/completions", request(tools=[]))
            self.assertEqual(status, 400)
            self.assertIn("Unsupported", json.loads(raw)["error"]["message"])
            self.assertEqual(len(service.runtime.contexts), 1)

    def test_stream_error_is_not_success_and_context_is_discarded(self):
        with running_server() as (port, service):
            service.runtime.fail = True
            with self.assertLogs("litert_server.server", level="ERROR"):
                status, raw = self.send(port, "/v1/chat/completions", request(stream=True))
            self.assertEqual(status, 200)
            self.assertIn('"error"', raw)
            self.assertNotIn("[DONE]", raw)
            self.assertFalse(service.slots)

    def test_existing_cat_agent_client_json_and_stream_continuation(self):
        # Exercise the unmodified real client, not another hand-built payload.
        source = str(Path(__file__).resolve().parents[2] / "src")
        with patch.object(sys, "path", [source, *sys.path]):
            from openai_agent.model_client import OpenAICompatibleChatClient
        with running_server() as (port, service), patch.dict(os.environ, {"CAT_AGENT_OPENAI_STREAM": "1"}):
            client = OpenAICompatibleChatClient(
                api_base_url=f"http://127.0.0.1:{port}/v1", model="test",
                timeout_seconds=5, retries=0, retry_delay_seconds=0,
                max_output_tokens=128000, temperature=0.0, top_p=1.0,
                reasoning_effort="none", label="test",
            )
            self.addCleanup(client.close)
            first = client.chat(request()["messages"])
            self.assertEqual(first.content, "Привет!")
            self.assertEqual(first.cached_tokens, 0)
            chunks = []
            client.set_event_handler(lambda label, event, payload: chunks.append(payload)
                                     if event == "chunk" else None)
            second = client.chat(request()["messages"])
            self.assertEqual(second.cached_tokens, 100)
            self.assertEqual(second.prompt_evaluated_tokens, 5)
            self.assertEqual("".join(chunks), "Привет!")
            history = request()["messages"] + [{"role": "assistant", "content": second.content}]
            third = client.chat(request("продолжай", history=history)["messages"])
            self.assertEqual(third.cached_tokens, 108)
            self.assertTrue(service.runtime.contexts[0].calls[-1][1])


# Fake C boundary checks the real TextContext orchestration, including rendering
# foreign history and checkpoint rewind. No model or native package is required.
class CFunction:
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, *args):
        return self.fn(*args)


class AdapterTests(unittest.TestCase):
    def test_base_is_prefilled_once_then_rewound_or_extended(self):
        sessions = {}
        def checkpoint(ptr, label):
            sessions[ptr].checkpoint = sessions[ptr].buffer
            return 0
        def rewind(ptr, label):
            sessions[ptr].buffer = sessions[ptr].checkpoint
            return 0
        def render(messages):
            return "<bos>" + "".join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        lib = SimpleNamespace(
            litert_lm_conversation_render_preface_to_string=CFunction(lambda ptr: ptr.preface.encode()),
            litert_lm_session_save_checkpoint=CFunction(checkpoint),
            litert_lm_session_rewind_to_checkpoint=CFunction(rewind),
        )
        class Renderer:
            def __init__(self, messages):
                self._lib, self._ptr = lib, self
                self.preface = render(messages)
                self.closed = False
            def render_message_to_string(self, message):
                return self.preface + render([message])[5:] + "<assistant>"
            def close(self):
                self.closed = True
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.close()
        class Session:
            def __init__(self):
                self._ptr = id(self)
                sessions[self._ptr] = self
                self.buffer, self.inputs = "", []
            def run_prefill(self, contents):
                self.inputs.extend(contents)
                self.buffer += "".join(contents)
                self.count = len("".join(contents))
            def get_benchmark_info(self):
                return SimpleNamespace(last_prefill_token_count=self.count, last_decode_token_count=2)
            def run_decode_async(self):
                self.buffer += "ok<end>"
                yield SimpleNamespace(texts=["ok"])
            def close(self):
                pass
        engine = SimpleNamespace(create_conversation=lambda **kw: Renderer(kw["messages"]),
                                 create_session=lambda **kw: Session())
        api = SimpleNamespace(SamplerConfig=lambda **kw: kw, ThinkingConfig=lambda **kw: kw)
        context = TextContext(api, engine, [BASE], {"sampler": {}, "max_output_tokens": 20})
        self.addCleanup(context.close)
        initial = request()["messages"]
        context.prefill(initial, False)
        list(context.decode())
        context.finish()
        history = initial + [{"role": "assistant", "content": "ok"}]
        continuation = request("next", history=history)["messages"]
        context.prefill(continuation, True)
        self.assertEqual(context.session.buffer.count(BASE["content"]), 1)
        self.assertIn("ok<end>", context.session.buffer)
        # New independent request must not retain the previous dialogue.
        context.prefill(request("fresh")["messages"], False)
        self.assertNotIn("ok<end>", context.session.buffer)
        self.assertNotIn("next", context.session.buffer)
        # Foreign/edited full history must be restored, never silently omitted.
        edited = [BASE, {"role": "user", "content": "old"},
                  {"role": "assistant", "content": "edited answer"},
                  {"role": "user", "content": "question"}]
        context.prefill(edited, False)
        self.assertIn("edited answer", context.session.buffer)
        self.assertNotIn("fresh", context.session.buffer)
        self.assertEqual(sum(BASE["content"] in text for text in context.session.inputs), 1)


if __name__ == "__main__":
    unittest.main()
