"""LiteRT-LM adapter. No imports from cat-agent or its configuration.

The Session checkpoint approach is also used by src/litert_agent/model_client.py.
Keep the three private C API calls here: their signatures are version-sensitive.
"""
from __future__ import annotations

import ctypes
import importlib.metadata
import logging
import os
from pathlib import Path
import time

LOG = logging.getLogger(__name__)
SUPPORTED_VERSIONS = ("0.16.0", "0.16.1", "0.18.0")


class LiteRTRuntime:
    def __init__(self, model_id, config):
        version = importlib.metadata.version("litert-lm")
        if version.split("+", 1)[0] not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"LiteRT-LM {version}: this adapter supports {', '.join(SUPPORTED_VERSIONS)}; "
                "verify the checkpoint C ABI before using another version"
            )
        import litert_lm

        self.api = litert_lm
        model_path = config.get("model_path")
        if not model_path:
            # Same registry layout as the CLI; no import from CLI internals.
            registry = Path(os.environ.get("LITERT_LM_DIR", Path.home() / ".litert-lm"))
            model_path = registry / "models" / model_id.replace("/", "--") / "model.litertlm"
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}; set model_path in config")
        backend = config.get("backend", "cpu")
        if backend == "cpu":
            backend = litert_lm.Backend.CPU(thread_count=config.get("cpu_thread_count"))
        elif backend == "gpu":
            backend = litert_lm.Backend.GPU()
        else:
            raise ValueError("backend must be cpu or gpu")
        started = time.monotonic()
        # 0.18 uses None to select the runtime's YNNPACK default. Do not turn
        # it off merely because an existing serve config omits this option.
        engine_options = {}
        if "enable_ynnpack" in config:
            engine_options["enable_ynnpack"] = config["enable_ynnpack"]
        self.engine = litert_lm.Engine(
            str(model_path), backend=backend,
            max_num_tokens=config.get("max_num_tokens", 8192),
            enable_speculative_decoding=config.get("speculative_decoding", False),
            enable_benchmark=True,
            # Rewinding must preserve the base, including local attention KV.
            use_ringbuffers_local_attention=False,
            cache_dir={"disk": "", "memory": ":memory", "no": ":nocache"}[
                config.get("cache", "disk")
            ],
            **engine_options,
        )
        LOG.info("ENGINE_READY model=%s version=%s seconds=%.3f", model_id, version,
                 time.monotonic() - started)

    def new_context(self, base, options):
        return TextContext(self.api, self.engine, base, options)

    def close(self):
        self.engine.close()


class TextContext:
    """One base checkpoint and one evolving dialogue; caller serializes access."""
    CHECKPOINT = b"litert-server-base"

    def __init__(self, api, engine, base, options):
        self.api, self.engine = api, engine
        self.base, self.options = base, options
        self.renderer = self.session = None
        self.resident = self.base_tokens = 0
        started = time.monotonic()
        LOG.info("BASE_WARM_BEGIN")
        try:
            self.renderer = self._renderer(base)
            self.lib = self.renderer._lib
            self.lib.litert_lm_conversation_render_preface_to_string.argtypes = [ctypes.c_void_p]
            self.lib.litert_lm_conversation_render_preface_to_string.restype = ctypes.c_char_p
            for name in ("save_checkpoint", "rewind_to_checkpoint"):
                fn = getattr(self.lib, "litert_lm_session_" + name)
                fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
                fn.restype = ctypes.c_int
            raw = self.lib.litert_lm_conversation_render_preface_to_string(self.renderer._ptr)
            if raw is None:
                raise RuntimeError("LiteRT could not render the base preface")
            self.preface = raw.decode("utf-8")
            self.session = engine.create_session(
                apply_prompt_template=False,
                sampler_config=api.SamplerConfig(**options["sampler"]),
                max_output_tokens=options["max_output_tokens"],
            )
            if self.preface:
                self.session.run_prefill([self.preface])
                self.base_tokens = self.session.get_benchmark_info().last_prefill_token_count
            self.resident = self.base_tokens
            self._checkpoint("save_checkpoint")
            LOG.info("BASE_WARM_READY tokens=%d seconds=%.3f", self.base_tokens,
                     time.monotonic() - started)
        except BaseException:
            self.close()
            raise

    def _renderer(self, messages):
        return self.engine.create_conversation(
            messages=messages, automatic_tool_calling=False,
            sampler_config=self.api.SamplerConfig(**self.options["sampler"]),
            thinking_config=self.api.ThinkingConfig(enable_thinking=False),
            extra_context={"enable_thinking": False},
        )

    def _checkpoint(self, operation):
        result = getattr(self.lib, "litert_lm_session_" + operation)(
            self.session._ptr, self.CHECKPOINT
        )
        if result != 0:
            raise RuntimeError(f"LiteRT {operation} failed: status={result}")

    def prefill(self, messages, continuation):
        """Append one new turn, or replay changed history after the cached base."""
        if continuation:
            rendered = self.renderer.render_message_to_string(messages[-1])
        else:
            self._checkpoint("rewind_to_checkpoint")
            self.resident = self.base_tokens
            if messages[:-1] == self.base:
                rendered = self.renderer.render_message_to_string(messages[-1])
            else:
                # Renderer formats existing assistant messages without generating them.
                with self._renderer(messages[:-1]) as renderer:
                    rendered = renderer.render_message_to_string(messages[-1])
        if not rendered.startswith(self.preface):
            raise RuntimeError("Chat template changed the base prefix; cannot reuse its KV safely")
        suffix = rendered[len(self.preface):]
        if not suffix:
            raise RuntimeError("Chat template produced an empty user turn")
        cached = self.resident
        self.session.run_prefill([suffix])
        evaluated = self.session.get_benchmark_info().last_prefill_token_count
        self.resident += evaluated
        return cached, evaluated

    def decode(self):
        for response in self.session.run_decode_async():
            if not response.texts or not isinstance(response.texts[0], str):
                raise RuntimeError("LiteRT returned invalid text")
            yield response.texts[0]

    def finish(self):
        count = self.session.get_benchmark_info().last_decode_token_count
        self.resident += count
        return count

    def close(self):
        for name in ("session", "renderer"):
            resource = getattr(self, name, None)
            if resource is not None:
                resource.close()
                setattr(self, name, None)
