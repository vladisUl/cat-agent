from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from orchestration.model_client import OpenAIChatClient


class _Handler(BaseHTTPRequestHandler):
    requests: list[tuple[str, dict]] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append((self.path, payload))

        if self.path == "/apply-template":
            body = json.dumps({"prompt": "<templated>test</templated>"}).encode()
        elif self.path == "/completion":
            body = json.dumps(
                {
                    "content": "ls test.txt",
                    "tokens_evaluated": 20,
                    "tokens_predicted": 4,
                    "timings": {
                        "prompt_n": 3,
                        "prompt_ms": 123.0,
                        "predicted_n": 4,
                        "predicted_ms": 456.0,
                    },
                }
            ).encode()
        else:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class OpenAIChatClientTest(unittest.TestCase):
    def setUp(self) -> None:
        _Handler.requests.clear()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.client = OpenAIChatClient(
            api_base_url=f"http://{host}:{port}/v1",
            model="gemma4-e4b",
            timeout_seconds=5,
            retries=0,
            retry_delay_seconds=0,
            max_output_tokens=64,
            temperature=0.0,
            top_p=1.0,
            reasoning_effort="none",
            id_slot=1,
        )

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_native_text_chat_without_tools(self) -> None:
        response = self.client.chat([{"role": "user", "content": "test"}])
        self.assertEqual(response.content, "ls test.txt")
        self.assertEqual(response.prompt_tokens, 20)
        self.assertEqual(response.completion_tokens, 4)
        self.assertEqual(response.cached_tokens, 17)
        self.assertEqual(response.prompt_evaluated_tokens, 3)
        self.assertEqual(response.prompt_seconds, 0.123)
        self.assertEqual(response.generation_seconds, 0.456)

        self.assertEqual(len(_Handler.requests), 2)
        template_path, template_payload = _Handler.requests[0]
        self.assertEqual(template_path, "/apply-template")
        self.assertEqual(template_payload["messages"], [{"role": "user", "content": "test"}])

        completion_path, payload = _Handler.requests[1]
        self.assertEqual(completion_path, "/completion")
        self.assertEqual(payload["prompt"], "<templated>test</templated>")
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["n_predict"], 64)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["id_slot"], 1)
        self.assertIs(payload["cache_prompt"], True)
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)


if __name__ == "__main__":
    unittest.main()
