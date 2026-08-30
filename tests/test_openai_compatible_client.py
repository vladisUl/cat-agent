from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

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

        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "REPLY\nОК",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 80},
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
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
