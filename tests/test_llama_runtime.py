from __future__ import annotations

import os
from unittest import mock
import unittest

from orchestration.assistant_manager import AssistantManagerRuntime
from orchestration.config import Settings
from llama_agent.model_client import LlamaChatClient
from llama_agent.runtime import AGENT_SLOT, MANAGER_SLOT, build_bundle


class FakeLlamaClient(LlamaChatClient):
    def __init__(self) -> None:
        super().__init__(
            "http://127.0.0.1:9380/v1",
            "/tmp/model.gguf",
            30,
            0,
            0.0,
            64,
            0.0,
            1.0,
            "none",
            label="test",
            id_slot=1,
        )
        self.requests: list[tuple[int, bool]] = []

    def _apply_template(self, messages):
        return "PROMPT:" + repr(messages)

    def _tokenize(self, prompt):
        return list(range(17))

    def _completion_request(self, prompt, *, n_predict, stream):
        self.requests.append((n_predict, stream))
        if n_predict == 0:
            return {
                "content": "",
                "tokens_evaluated": 17,
                "tokens_predicted": 0,
                "tokens_cached": 0,
                "timings": {"prompt_ms": 100.0, "predicted_ms": 0.0},
            }
        return {
            "content": "REPLY\nOK",
            "tokens_evaluated": 21,
            "tokens_predicted": 5,
            "tokens_cached": 17,
            "timings": {
                "prompt_n": 4,
                "prompt_ms": 40.0,
                "predicted_n": 5,
                "predicted_ms": 50.0,
            },
        }


class LlamaClientTest(unittest.TestCase):
    def test_prepare_prefix_really_prefills_slot(self) -> None:
        client = FakeLlamaClient()
        base = [{"role": "system", "content": "BASE"}]

        warm = client.prepare_prefix(base)

        self.assertEqual(client.requests, [(0, False)])
        self.assertEqual(warm.token_count, 17)
        self.assertEqual(client.resident_tokens, 17)

    def test_reset_to_base_does_not_make_an_extra_model_request(self) -> None:
        client = FakeLlamaClient()
        base = [{"role": "system", "content": "BASE"}]
        client.prepare_prefix(base)
        client.requests.clear()

        client.reset_to_base(base)

        self.assertEqual(client.requests, [])
        self.assertEqual(client.resident_tokens, 17)

    def test_chat_reports_llama_cache_reuse(self) -> None:
        client = FakeLlamaClient()
        base = [{"role": "system", "content": "BASE"}]
        client.prepare_prefix(base)

        response = client.chat(base + [{"role": "user", "content": "hello"}])

        self.assertEqual(response.content, "REPLY\nOK")
        self.assertEqual(response.cached_tokens, 17)
        self.assertEqual(response.prompt_evaluated_tokens, 4)
        self.assertEqual(response.completion_tokens, 5)
        self.assertEqual(client.resident_tokens, 26)


class LlamaRuntimeTest(unittest.TestCase):
    def test_runtime_uses_current_assistant_manager_and_two_llama_slots(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "CAT_AGENT_MODEL": "/storage/models/test.gguf",
                "CAT_AGENT_AGENT_COUNT": "3",
            },
            clear=False,
        ):
            settings = Settings.from_env()
            bundle = build_bundle(settings)

        try:
            self.assertIsInstance(bundle.runtime, AssistantManagerRuntime)
            self.assertEqual(bundle.manager_client.id_slot, MANAGER_SLOT)
            self.assertEqual(bundle.agent_client.id_slot, AGENT_SLOT)
            workers = [bundle.runtime.pool.get(f"agent{index}") for index in range(1, 4)]
            self.assertTrue(all(worker is not None for worker in workers))
            for worker in workers:
                assert worker is not None
                self.assertIs(worker.client, bundle.agent_client)
        finally:
            bundle.close()


if __name__ == "__main__":
    unittest.main()
