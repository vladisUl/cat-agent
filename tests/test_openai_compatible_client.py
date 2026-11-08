from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import threading
import unittest
from unittest import mock

from openai_agent.model_client import OpenAICompatibleChatClient


class _Handler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict, str | None]] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(
            (self.path, payload, self.headers.get("Authorization"))
        )
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return

        if payload["model"] == "trace-http-500":
            body = b'{"error":{"message":"Operation failed"}}'
            status, content_type = 500, "application/json"
        elif payload["stream"]:
            events = [
                {"choices": [{"delta": {"reasoning_content": "Сначала прочитаю запрос."}}]},
                {"choices": [{"delta": {"content": "REPLY\nОК"}}]},
                {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 8,
                    "completion_tokens_details": {"reasoning_tokens": 5}}},
            ]
            body = ("".join("data: " + json.dumps(e, ensure_ascii=False) + "\n\n"
                            for e in events) + "data: [DONE]\n\n").encode()
            status, content_type = 200, "text/event-stream"
        else:
            body = json.dumps(
                {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "REPLY\nОК",
                            "reasoning_content": "Сначала прочитаю запрос.",
                        },
                    }],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 7,
                        "prompt_tokens_details": {"cached_tokens": 80},
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    },
                },
                ensure_ascii=False,
            ).encode()
            status, content_type = 200, "application/json"
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class OpenAICompatibleChatClientTest(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.requests.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = OpenAICompatibleChatClient(
            api_base_url=f"http://{host}:{port}/v1",
            model="test-model",
            timeout_seconds=5,
            retries=0,
            retry_delay_seconds=0,
            max_output_tokens=64,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort="none",
            label="manager",
            api_key="secret",
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _trace_client(self, model="test-model"):
        with mock.patch.dict(os.environ, {"CAT_AGENT_OPENAI_TRACE": "1"}):
            client = OpenAICompatibleChatClient(
                api_base_url=self.client.api_base_url, model=model,
                timeout_seconds=5, retries=0, retry_delay_seconds=0,
                max_output_tokens=64, temperature=0.0, top_p=1.0,
                reasoning_effort="none", label="manager",
                api_key="DO_NOT_LOG_THIS_KEY",
                extra_body={"thinking": {"type": "disabled"}},
            )
        self.assertTrue(client.trace_enabled)
        self.assertTrue(client.fork("agent1").trace_enabled)
        return client

    def test_trace_non_stream_logs_raw_reasoning_tokens_and_request(self) -> None:
        client = self._trace_client()
        with self.assertLogs("openai_agent.model_client", level="INFO") as logs:
            response = client.chat([{"role": "user", "content": "секретный вопрос"}])
        self.assertEqual(response.content, "REPLY\nОК")
        combined = "\n".join(logs.output)
        self.assertIn('"thinking": {"type": "disabled"}', combined)
        self.assertIn("секретный вопрос", combined)
        self.assertIn("Сначала прочитаю запрос.", combined)
        self.assertIn('"reasoning_tokens": 3', combined)
        self.assertIn("reasoning_content_present=True", combined)
        self.assertIn("status=200", combined)
        self.assertNotIn("DO_NOT_LOG_THIS_KEY", combined)

    def test_trace_stream_logs_sse_reasoning(self) -> None:
        client = self._trace_client()
        client.stream_enabled = True
        client.set_event_handler(lambda *args: None)
        with self.assertLogs("openai_agent.model_client", level="INFO") as logs:
            response = client.chat([{"role": "user", "content": "привет"}])
        combined = "\n".join(logs.output)
        self.assertEqual(response.content, "REPLY\nОК")
        self.assertIn("Сначала прочитаю запрос.", combined)
        self.assertIn('"reasoning_tokens": 5', combined)
        self.assertIn("reasoning_content_present=True", combined)
        self.assertIn("stream=True", combined)
        self.assertNotIn("DO_NOT_LOG_THIS_KEY", combined)

    def test_trace_logs_raw_http_error(self) -> None:
        from orchestration.model_client import ModelClientError
        client = self._trace_client(model="trace-http-500")
        with self.assertLogs("openai_agent.model_client", level="INFO") as logs:
            with self.assertRaises(ModelClientError):
                client.chat([{"role": "user", "content": "привет"}])
        combined = "\n".join(logs.output)
        self.assertIn("status=500", combined)
        self.assertIn("Operation failed", combined)
        self.assertNotIn("DO_NOT_LOG_THIS_KEY", combined)

    def test_chat_completions_request(self) -> None:
        response = self.client.chat(
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "test"},
            ]
        )

        self.assertEqual(response.content, "REPLY\nОК")
        self.assertEqual(response.prompt_tokens, 100)
        self.assertEqual(response.completion_tokens, 7)
        self.assertEqual(response.cached_tokens, 80)
        self.assertEqual(response.prompt_evaluated_tokens, 20)
        self.assertEqual(self.client.resident_tokens, 0)

        self.assertEqual(len(_Handler.requests), 1)
        path, payload, authorization = _Handler.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(authorization, "Bearer secret")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["messages"][-1]["content"], "test")
        self.assertNotIn("reasoning_effort", payload)


if __name__ == "__main__":
    unittest.main()
