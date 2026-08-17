from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import ctypes
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


class LiteRTChatClient:
    """LiteRT-LM adapter backed by the low-level Session API."""

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

    @property
    def resident_tokens(self) -> int:
        return self._resident_tokens

    @property
    def last_response(self) -> ChatResponse | None:
        return self._last_response

    @property
    def warm_result(self) -> WarmResult | None:
        return self._warm_result

    def wait_until_ready(
        self, stop_requested: Callable[[], bool], interval: float = 2.0
    ) -> bool:
        del interval
        return not stop_requested()

    def close(self) -> None:
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

    def chat(self, messages: list[dict[str, str]]) -> ChatResponse:
        if not messages or messages[-1].get("role") != "user":
            raise ModelClientError("LiteRT Session turn must end in a user message")

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

        try:
            user_turn = self._render_user_turn(messages[-1])
            resident_before = self._resident_tokens
            total_started = time.monotonic()

            prefill_started = time.monotonic()
            self._session.run_prefill([user_turn])
            prompt_seconds = time.monotonic() - prefill_started
            prefill_benchmark = self._session.get_benchmark_info()
            prefill_n = prefill_benchmark.last_prefill_token_count

            decode_started = time.monotonic()
            response = self._session.run_decode()
            generation_seconds = time.monotonic() - decode_started
            decode_benchmark = self._session.get_benchmark_info()
            decode_n = decode_benchmark.last_decode_token_count

            elapsed = time.monotonic() - total_started
            content = _session_response_text(response)

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
            raise
        except Exception as exc:
            raise ModelClientError(f"LiteRT-LM Session request failed: {exc}") from exc

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
