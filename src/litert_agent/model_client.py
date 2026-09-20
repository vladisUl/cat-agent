from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import ctypes
from agent_core.metrics import measured
import json
import logging
import time
from typing import Callable

import litert_lm

from orchestration.model_client import ChatResponse, ModelClientError

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WarmResult:
    strategy: str
    elapsed_seconds: float
    token_count: int


from agent_core.types import InferenceTiming


ModelEventHandler = Callable[[str, str, str], None]


class LiteRTChatClient:
    supports_resident_context_pool = True

    """LiteRT-LM adapter backed by the low-level Session API."""

    supports_images = True

    def __init__(
        self,
        engine: litert_lm.Engine,
        *,
        max_output_tokens: int,
        temperature: float,
        top_p: float,
        reasoning_effort: str,
        label: str,
        allow_prefix_reset: bool = False,
    ) -> None:
        self._temperature = temperature
        self._top_p = top_p
        self.engine = engine
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.label = label
        self.allow_prefix_reset = allow_prefix_reset
        self.sampler_config = litert_lm.SamplerConfig(
            top_p=top_p,
            temperature=temperature,
            seed=0,
        )
        self._renderer: litert_lm.Conversation | None = None
        self._vision_conversation = None
        self._session = None
        self._lib = None
        self._base_preface = ""
        self._base_messages: list[dict[str, str]] = []
        self._base_resident_tokens = 0
        self._checkpoint_label = f"cat-agent-{label}-base".encode("utf-8")
        self._checkpoint_ready = False
        self._synced_messages: list[dict[str, str]] = []
        self._resident_tokens = 0
        self._last_response: ChatResponse | None = None
        self._warm_result: WarmResult | None = None
        self._event_handler: ModelEventHandler | None = None
        self._inference_timing = InferenceTiming("idle", None, None, None, None, None)

    def fork(self, label):
        client = LiteRTChatClient(
            self.engine, max_output_tokens=self.max_output_tokens,
            temperature=self._temperature, top_p=self._top_p,
            reasoning_effort=self.reasoning_effort, label=label, allow_prefix_reset=True,
        )
        if self._base_messages:
            client.prepare_prefix(self._base_messages)
        return client

    @property
    def resident_tokens(self) -> int:
        return self._resident_tokens

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    @property
    def warm_result(self) -> WarmResult | None:
        return self._warm_result

    @property
    def inference_timing(self) -> InferenceTiming:
        return self._inference_timing

    def set_event_handler(self, handler: ModelEventHandler | None) -> None:
        self._event_handler = handler

    def wait_until_ready(
        self, stop_requested: Callable[[], bool], interval: float = 2.0
    ) -> bool:
        del interval
        return not stop_requested()

    def close(self) -> None:
        self._close_vision()
        if self._session is not None:
            self._session.close()
            self._session = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._lib = None
        self._base_preface = ""
        self._base_messages = []
        self._base_resident_tokens = 0
        self._checkpoint_ready = False
        self._synced_messages = []
        self._resident_tokens = 0
        self._last_response = None
        self._warm_result = None
        self._inference_timing = InferenceTiming("idle", None, None, None, None, None)

    def prepare_prefix(self, messages: list[dict[str, str]]) -> WarmResult:
        if (
            len(messages) != 1
            or messages[0].get("role") != "system"
            or not messages[0].get("content", "").strip()
        ):
            raise ModelClientError(
                "LiteRT Session base must be exactly one non-empty system message"
            )

        try:
            self.close()
            self._renderer = self.engine.create_conversation(
                messages=deepcopy(messages),
                automatic_tool_calling=False,
                sampler_config=self.sampler_config,
                max_output_tokens=self.max_output_tokens,
            )
            self._lib = self._renderer._lib
            self._configure_renderer_ffi()
            self._configure_session_checkpoint_ffi()

            raw = self._lib.litert_lm_conversation_render_preface_to_string(
                self._renderer._ptr
            )
            if not raw:
                raise RuntimeError("render_preface_to_string failed")
            self._base_preface = raw.decode("utf-8")

            self._session = self.engine.create_session(
                apply_prompt_template=False,
                sampler_config=self.sampler_config,
                max_output_tokens=self.max_output_tokens,
            )

            started = time.monotonic()
            self._session.run_prefill([self._base_preface])
            elapsed = time.monotonic() - started

            benchmark = self._session.get_benchmark_info()
            prefill_n = benchmark.last_prefill_token_count
            if prefill_n <= 0:
                raise RuntimeError(
                    f"Session prefix prefill returned {prefill_n} tokens"
                )

            self._base_messages = deepcopy(messages)
            self._base_resident_tokens = prefill_n
            self._resident_tokens = prefill_n
            self._synced_messages = deepcopy(messages)
            self._save_base_checkpoint()

            LOGGER.info(
                "litert-session-warm %s resident=%d elapsed=%.3fs checkpoint=%s",
                self.label,
                self._resident_tokens,
                elapsed,
                self._checkpoint_label.decode("utf-8"),
            )
            result = WarmResult(
                strategy="session-checkpoint",
                elapsed_seconds=elapsed,
                token_count=prefill_n,
            )
            self._warm_result = result
            return result
        except ModelClientError:
            raise
        except Exception as exc:
            self.close()
            raise ModelClientError(
                f"LiteRT-LM Session prefix prefill failed: {exc}"
            ) from exc

    def reset_to_base(self, messages: list[dict[str, str]]) -> None:
        if messages != self._base_messages:
            raise ModelClientError(
                f"LiteRT Session {self.label} requested reset does not match resident base"
            )
        if (
            self._session is None
            or self._lib is None
            or not self._checkpoint_ready
        ):
            raise ModelClientError(
                f"LiteRT Session {self.label} has no resident base checkpoint"
            )

        try:
            self._close_vision()
            result = self._lib.litert_lm_session_rewind_to_checkpoint(
                self._session._ptr,
                self._checkpoint_label,
            )
            if result != 0:
                raise RuntimeError(
                    f"rewind_to_checkpoint returned {result}"
                )
            self._resident_tokens = self._base_resident_tokens
            self._synced_messages = deepcopy(self._base_messages)
            self._last_response = None
            LOGGER.info(
                "litert-session-rewind %s resident=%d checkpoint=%s",
                self.label,
                self._resident_tokens,
                self._checkpoint_label.decode("utf-8"),
            )
        except ModelClientError:
            raise
        except Exception as exc:
            raise ModelClientError(
                f"LiteRT-LM Session rewind failed: {exc}"
            ) from exc

    @measured("model_seconds")
    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        if not messages or messages[-1].get("role") != "user":
            raise ModelClientError("LiteRT Session turn must end in a user message")

        if self._vision_conversation is not None and messages[:-1] != self._synced_messages:
            if not self._try_reset_prefix(messages):
                raise ModelClientError("LiteRT vision history no longer extends the current conversation")
        if self._vision_conversation is not None or isinstance(messages[-1].get("content"), list):
            return self._chat_multimodal(messages)

        if self._session is None or self._renderer is None:
            if not self._try_reset_prefix(messages):
                raise ModelClientError(
                    "LiteRT Session client has not been prepared with prepare_prefix()"
                )

        if messages[:-1] != self._synced_messages:
            if not self._try_reset_prefix(messages):
                raise ModelClientError(
                    f"LiteRT Session {self.label} history no longer extends resident KV"
                )

        total_started = time.monotonic()
        prompt_seconds: float | None = None
        generation_seconds: float | None = None
        try:
            user_turn = self._render_user_turn(messages[-1])
            resident_before = self._resident_tokens

            prefill_started = time.monotonic()
            self._inference_timing = InferenceTiming(
                "prefill", prefill_started, None, None, None, None
            )
            self._emit_event("prefill_start")
            self._session.run_prefill([user_turn])
            prompt_seconds = time.monotonic() - prefill_started
            prefill_benchmark = self._session.get_benchmark_info()
            prefill_n = prefill_benchmark.last_prefill_token_count

            decode_started = time.monotonic()
            self._inference_timing = InferenceTiming(
                "generate", decode_started, prompt_seconds, None, None, None
            )
            self._emit_event("decode_start")

            chunks: list[str] = []
            run_decode_async = getattr(self._session, "run_decode_async", None)
            if self._event_handler is not None and callable(run_decode_async):
                for chunk_response in run_decode_async():
                    chunk = _session_response_text(chunk_response)
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    self._emit_event("chunk", chunk)
                content = "".join(chunks)
                if not content:
                    raise ModelClientError("LiteRT Session streamed decode returned no text")
            else:
                response = self._session.run_decode()
                content = _session_response_text(response)
                self._emit_event("chunk", content)

            generation_seconds = time.monotonic() - decode_started
            decode_benchmark = self._session.get_benchmark_info()
            decode_n = decode_benchmark.last_decode_token_count

            elapsed = time.monotonic() - total_started
            finished_at = time.monotonic()
            self._inference_timing = InferenceTiming(
                "idle",
                None,
                prompt_seconds,
                generation_seconds,
                elapsed,
                finished_at,
            )
            self._emit_event("decode_done")

            self._resident_tokens += prefill_n + decode_n
            self._synced_messages = deepcopy(messages)
            self._synced_messages.append(
                {"role": "assistant", "content": content}
            )

            LOGGER.info(
                "litert-session-kv %s resident=%d new=%d decode=%d "
                "after=%d wall=%.3fs prefill=%.3fs generate=%.3fs",
                self.label,
                resident_before,
                prefill_n,
                decode_n,
                self._resident_tokens,
                elapsed,
                prompt_seconds,
                generation_seconds,
            )

            result = ChatResponse(
                content=content,
                prompt_tokens=resident_before + prefill_n,
                completion_tokens=decode_n,
                elapsed_seconds=elapsed,
                cached_tokens=resident_before,
                prompt_evaluated_tokens=prefill_n,
                prompt_seconds=prompt_seconds,
                generation_seconds=generation_seconds,
            )
            self._last_response = result
            return result
        except ModelClientError:
            self._finish_failed_timing(total_started, prompt_seconds, generation_seconds)
            raise
        except Exception as exc:
            self._finish_failed_timing(total_started, prompt_seconds, generation_seconds)
            raise ModelClientError(f"LiteRT-LM Session request failed: {exc}") from exc

    def _close_vision(self):
        if self._vision_conversation is not None:
            self._vision_conversation.close()
            self._vision_conversation = None

    def _chat_multimodal(self, messages):
        """Move the complete logical dialogue to the native vision processor.

        Session.run_prefill in LiteRT's Python API accepts text only. Conversation
        owns model-specific image preprocessing; keep it for all subsequent turns
        until reset_to_base, rather than starting an independent image query.
        """
        started = time.monotonic()
        first_chunk = None
        self._inference_timing = InferenceTiming("prefill", started, None, None, None, None)
        self._emit_event("prefill_start")
        try:
            if self._vision_conversation is None:
                self._vision_conversation = self.engine.create_conversation(
                    messages=[_vision_message(m) for m in messages[:-1]],
                    automatic_tool_calling=False,
                    sampler_config=self.sampler_config,
                    max_output_tokens=self.max_output_tokens,
                )
                LOGGER.info("litert-vision-context %s migrated_messages=%d", self.label, len(messages) - 1)
            conversation = self._vision_conversation
            chunks = []
            for response in conversation.send_message_async(_vision_message(messages[-1])):
                content = response.get("content", [])
                text = content if isinstance(content, str) else "".join(
                    part.get("text", "") for part in content if part.get("type") == "text"
                )
                if not text:
                    continue
                if first_chunk is None:
                    first_chunk = time.monotonic()
                    self._inference_timing = InferenceTiming(
                        "generate", first_chunk, first_chunk - started, None, None, None
                    )
                    self._emit_event("decode_start")
                chunks.append(text)
                self._emit_event("chunk", text)
            text = "".join(chunks)
            if not text:
                raise ModelClientError("LiteRT vision conversation returned no text")
            finished = time.monotonic()
            elapsed = finished - started
            # TTFT includes image processing, prefill and the first decoded token.
            prompt_seconds = first_chunk - started
            generation_seconds = finished - first_chunk
            benchmark = conversation.get_benchmark_info()
            prefill_n = benchmark.last_prefill_token_count
            decode_n = benchmark.last_decode_token_count
            # The public benchmark exposes only the last prefill, not the full
            # migrated history/vision KV size. Do not report an invented total.
            self._resident_tokens = None
            self._synced_messages = deepcopy(messages)
            self._synced_messages.append({"role": "assistant", "content": text})
            result = ChatResponse(
                content=text, prompt_tokens=None, completion_tokens=decode_n,
                elapsed_seconds=elapsed, cached_tokens=None,
                prompt_evaluated_tokens=prefill_n, prompt_seconds=prompt_seconds,
                generation_seconds=generation_seconds,
            )
            self._last_response = result
            self._inference_timing = InferenceTiming(
                "idle", None, prompt_seconds, generation_seconds, elapsed, finished
            )
            self._emit_event("decode_done")
            LOGGER.info("litert-vision %s wall=%.3fs ttft=%.3fs", self.label, elapsed, prompt_seconds)
            return result
        except Exception as exc:
            self._close_vision()
            self._synced_messages = []  # Never resume the stale text KV after a failed image turn.
            self._finish_failed_timing(started, None, None)
            raise ModelClientError(f"LiteRT-LM vision request failed: {exc}") from exc

    def _finish_failed_timing(
        self,
        total_started: float,
        prompt_seconds: float | None,
        generation_seconds: float | None,
    ) -> None:
        now = time.monotonic()
        current = self._inference_timing
        if prompt_seconds is None and current.phase == "prefill" and current.phase_started is not None:
            prompt_seconds = now - current.phase_started
        if generation_seconds is None and current.phase == "generate" and current.phase_started is not None:
            generation_seconds = now - current.phase_started
        self._inference_timing = InferenceTiming(
            "idle",
            None,
            prompt_seconds,
            generation_seconds,
            now - total_started,
            now,
        )
        self._emit_event("decode_error")

    def _emit_event(self, event: str, payload: str = "") -> None:
        handler = self._event_handler
        if handler is None:
            return
        try:
            handler(self.label, event, payload)
        except Exception:
            LOGGER.debug("model event handler failed label=%s event=%s", self.label, event, exc_info=True)

    def _try_reset_prefix(self, messages: list[dict[str, str]]) -> bool:
        if not self.allow_prefix_reset or len(messages) != 2:
            return False
        prefix = messages[:1]
        if prefix[0].get("role") != "system":
            return False
        if self._checkpoint_ready and prefix == self._base_messages:
            self.reset_to_base(prefix)
        else:
            self.prepare_prefix(prefix)
        return messages[:-1] == self._synced_messages

    def _configure_renderer_ffi(self) -> None:
        assert self._lib is not None

        self._lib.litert_lm_conversation_render_preface_to_string.argtypes = [
            ctypes.c_void_p
        ]
        self._lib.litert_lm_conversation_render_preface_to_string.restype = ctypes.c_char_p
        self._lib.litert_lm_conversation_render_message_to_string.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._lib.litert_lm_conversation_render_message_to_string.restype = ctypes.c_char_p

    def _configure_session_checkpoint_ffi(self) -> None:
        assert self._lib is not None

        required = (
            "litert_lm_session_save_checkpoint",
            "litert_lm_session_rewind_to_checkpoint",
        )
        missing = [name for name in required if not hasattr(self._lib, name)]
        if missing:
            raise RuntimeError(
                "installed LiteRT-LM lacks Session checkpoint C API; "
                "LiteRT-LM 0.16.0 or newer is required: "
                + ", ".join(missing)
            )

        self._lib.litert_lm_session_save_checkpoint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._lib.litert_lm_session_save_checkpoint.restype = ctypes.c_int
        self._lib.litert_lm_session_rewind_to_checkpoint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._lib.litert_lm_session_rewind_to_checkpoint.restype = ctypes.c_int

    def _save_base_checkpoint(self) -> None:
        assert self._session is not None
        assert self._lib is not None

        result = self._lib.litert_lm_session_save_checkpoint(
            self._session._ptr,
            self._checkpoint_label,
        )
        if result != 0:
            raise RuntimeError(f"save_checkpoint returned {result}")
        self._checkpoint_ready = True

    def _render_user_turn(self, message: dict[str, str]) -> str:
        assert self._renderer is not None
        assert self._lib is not None

        message_json = json.dumps(message, ensure_ascii=False).encode("utf-8")
        raw = self._lib.litert_lm_conversation_render_message_to_string(
            self._renderer._ptr,
            message_json,
        )
        if not raw:
            raise RuntimeError("render_message_to_string failed")

        rendered = raw.decode("utf-8")
        if not rendered.startswith(self._base_preface):
            raise RuntimeError(
                "Rendered user turn does not extend canonical base preface"
            )

        suffix = rendered[len(self._base_preface) :]
        if not suffix:
            raise RuntimeError("Rendered user turn is empty")
        return suffix


def _vision_message(message):
    """Translate the shared OpenAI content representation to LiteRT JSON."""
    result = deepcopy(message)
    content = result.get("content")
    if isinstance(content, list):
        converted = []
        for part in content:
            if part.get("type") == "text":
                converted.append(part)
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                header, separator, encoded = url.partition(",")
                if not separator or header not in {"data:image/png;base64", "data:image/jpeg;base64"}:
                    raise ModelClientError("LiteRT images must be embedded PNG/JPEG data")
                converted.append({"type": "image", "blob": encoded})
            else:
                raise ModelClientError("Unsupported multimodal content type")
        result["content"] = converted
    return result


def _session_response_text(response) -> str:
    texts = getattr(response, "texts", None)
    if not texts:
        raise ModelClientError(f"LiteRT Session decode returned no text: {response!r}")

    text = texts[0]
    if not isinstance(text, str):
        raise ModelClientError(
            f"LiteRT Session decode returned invalid text: {text!r}"
        )
    return text

