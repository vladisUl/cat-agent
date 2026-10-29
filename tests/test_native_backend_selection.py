"""CPU/GPU Engine selection tests without loading a LiteRT model."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from orchestration_tools.litert_runtime import build_native_manager


class EngineCreated(Exception):
    """Stop immediately after recording the native Engine configuration."""


class NativeBackendSelectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = Path(temp.name) / "model.litertlm"
        self.model.write_bytes(b"test-only")
        self.prompt = Path(temp.name) / "manager.md"
        self.prompt.write_text("Native manager", encoding="utf-8")
        self.calls = []

    def probe(self, **options):
        class FakeBackend:
            @staticmethod
            def CPU(thread_count=None):
                return ("cpu", thread_count)

            @staticmethod
            def GPU():
                return ("gpu", None)

        class FakeActivation:
            @staticmethod
            def from_str(value):
                return "dtype:" + value

        def create_engine(model, **kwargs):
            self.calls.append((model, kwargs))
            raise EngineCreated()

        fake_litert = SimpleNamespace(
            Engine=create_engine,
            Backend=FakeBackend,
            ActivationDataType=FakeActivation,
        )
        with patch.dict(sys.modules, {"litert_lm": fake_litert}):
            with self.assertRaises(EngineCreated):
                build_native_manager(
                    self.model, self.prompt, enable_mqtt=False,
                    **options,
                )
        self.assertEqual(len(self.calls), 1)
        model, kwargs = self.calls[0]
        self.assertEqual(model, str(self.model))
        return kwargs

    def test_cpu_keeps_known_good_settings(self):
        settings = self.probe(backend_name="cpu", cpu_threads=8)
        self.assertEqual(settings["backend"], ("cpu", 8))
        self.assertEqual(settings["vision_backend"], ("cpu", None))
        self.assertFalse(settings["enable_ynnpack"])
        self.assertFalse(settings["enable_speculative_decoding"])
        self.assertNotIn("activation_data_type", settings)

    def test_gpu_uses_gpu_and_profile_flags_without_cpu_fallback(self):
        settings = self.probe(
            backend_name="gpu", activation_dtype="fp32", speculative=True,
        )
        self.assertEqual(settings["backend"], ("gpu", None))
        self.assertEqual(settings["activation_data_type"], "dtype:fp32")
        self.assertTrue(settings["enable_speculative_decoding"])
        self.assertEqual(settings["vision_backend"], ("cpu", None))
        self.assertNotIn("enable_ynnpack", settings)

    def test_gpu_text_only_omits_vision_backend_entirely(self):
        settings = self.probe(
            backend_name="gpu", vision_backend_name="none",
            activation_dtype="fp32",
        )
        self.assertEqual(settings["backend"], ("gpu", None))
        self.assertNotIn("vision_backend", settings)
        self.assertNotIn("enable_ynnpack", settings)

    def test_gpu_with_explicit_gpu_vision_backend(self):
        settings = self.probe(
            backend_name="gpu", vision_backend_name="gpu",
        )
        self.assertEqual(settings["vision_backend"], ("gpu", None))

    def test_unknown_vision_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "vision backend"):
            build_native_manager(
                self.model, self.prompt, enable_mqtt=False,
                vision_backend_name="invalid",
            )

    def test_bad_backend_rejected_before_native_library_import(self):
        with self.assertRaisesRegex(ValueError, "backend must be cpu or gpu"):
            build_native_manager(self.model, self.prompt, enable_mqtt=False,
                                 backend_name="unconfigured")


if __name__ == "__main__":
    unittest.main()
