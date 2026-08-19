from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from litert_agent import runtime


class FakeEngine:
    calls: list[tuple[str, dict[str, object]]] = []

    def __init__(self, model_path: str, **kwargs: object) -> None:
        self.model_path = model_path
        self.kwargs = kwargs
        self.calls.append((model_path, dict(kwargs)))

    def __enter__(self):
        return self


class LiteRTRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeEngine.calls.clear()

    def test_create_engine_passes_ynnpack_option(self) -> None:
        backend = object()
        with (
            patch.object(runtime, "_backend", return_value=backend),
            patch.object(runtime.litert_lm, "Engine", FakeEngine),
        ):
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    FakeEngine.calls.clear()
                    runtime._create_engine(
                        Path("model.litertlm"),
                        "cpu",
                        8,
                        None,
                        False,
                        enabled,
                        None,
                        label="test",
                    )
                    self.assertEqual(len(FakeEngine.calls), 1)
                    model_path, kwargs = FakeEngine.calls[0]
                    self.assertEqual(model_path, "model.litertlm")
                    self.assertIs(kwargs["backend"], backend)
                    self.assertEqual(kwargs["enable_ynnpack"], enabled)


if __name__ == "__main__":
    unittest.main()
