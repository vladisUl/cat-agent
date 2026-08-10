from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from cat_agent.model_client import OpenAIChatClient


class _Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = json.dumps({"object": "list", "data": [{"id": "gemma4-e4b"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.__class__.requests.append(payload)
        response = {
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ls test.txt"},
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "total_tokens": 24,
                "prompt_tokens_details": {"cached_tokens": 17},
            },
            "timings": {
                "cache_n": 17,
                "prompt_n": 3,
                "prompt_ms": 123.0,
                "predicted_ms": 456.0,
            },
        }
        body = json.dumps(response).encode()
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

    def test_models_and_text_chat_without_native_tools(self) -> None:
        models = self.client.list_models()
        self.assertEqual(models["data"][0]["id"], "gemma4-e4b")

        response = self.client.chat([{"role": "user", "content": "test"}])
        self.assertEqual(response.content, "ls test.txt")
        self.assertEqual(response.prompt_tokens, 20)
        self.assertEqual(response.completion_tokens, 4)
        self.assertEqual(response.cached_tokens, 17)
        self.assertEqual(response.prompt_evaluated_tokens, 3)
        self.assertEqual(response.prompt_seconds, 0.123)
        self.assertEqual(response.generation_seconds, 0.456)

        payload = _Handler.requests[0]
        self.assertEqual(payload["model"], "gemma4-e4b")
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertEqual(payload["max_completion_tokens"], 64)
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertEqual(payload["id_slot"], 1)
        self.assertIs(payload["cache_prompt"], True)


if __name__ == "__main__":
    unittest.main()
