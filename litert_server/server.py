"""Local text/image inference server; serialized native inference over HTTP."""
from __future__ import annotations

import argparse
import base64
import binascii
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import threading
import time
import uuid

from .runtime import LiteRTRuntime

LOG = logging.getLogger(__name__)
MAX_BODY_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024


class RequestError(ValueError):
    pass


def positive_int(value, name):
    if type(value) is not int or value < 1:
        raise RequestError(f"{name} must be a positive integer")
    return value


def sampler_values(source):
    result = {}
    for name in ("temperature", "top_p", "top_k", "seed"):
        value = source.get(name)
        if value is None:
            continue
        if name in {"top_k", "seed"}:
            if type(value) is not int or (name == "top_k" and value < 1):
                raise RequestError(f"Invalid {name}")
        elif (type(value) not in (int, float) or not math.isfinite(value)
              or value < 0 or (name == "top_p" and value > 1)):
            raise RequestError(f"Invalid {name}")
        result[name] = value
    return result


def load_config(path, model_id=None):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data) - {"default", "models"}:
        raise ValueError("Config must contain only default and models objects")
    models = data.get("models", {})
    if not isinstance(models, dict) or not models:
        raise ValueError("Config needs at least one entry in models")
    if model_id is None:
        if len(models) != 1:
            raise ValueError("Use --model to select one configured model")
        model_id = next(iter(models))
    if not isinstance(model_id, str) or not model_id or model_id not in models:
        raise ValueError("Unknown configured model")
    defaults, selected = data.get("default", {}), models[model_id]
    if not isinstance(defaults, dict) or not isinstance(selected, dict):
        raise ValueError("default and model settings must be objects")
    config = {**defaults, **selected}
    allowed = {
        "model_path", "backend", "cpu_thread_count", "speculative_decoding",
        "enable_ynnpack", "cache", "max_num_tokens", "max_output_tokens",
        "temperature", "top_p", "top_k", "seed", "thinking", "vision_backend",
    }
    if set(config) - allowed:
        raise ValueError(f"Unsupported settings: {sorted(set(config) - allowed)}")
    for name in ("cpu_thread_count", "max_num_tokens", "max_output_tokens"):
        if name in config:
            positive_int(config[name], name)
    if config.get("max_num_tokens", 8192) < 2:
        raise ValueError("max_num_tokens must be at least 2")
    for name in ("speculative_decoding", "enable_ynnpack", "thinking"):
        if name in config and type(config[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if config.get("thinking", False):
        raise ValueError("This text Session adapter supports thinking=false only")
    if config.get("backend", "cpu") not in ("cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu")
    if "vision_backend" in config and config["vision_backend"] not in ("cpu", "gpu", "none"):
        raise ValueError("vision_backend must be cpu, gpu or none")
    if config.get("cache", "disk") not in ("disk", "memory", "no"):
        raise ValueError("cache must be disk, memory or no")
    sampler_values(config)
    if "model_path" in config:
        if not isinstance(config["model_path"], str) or not config["model_path"]:
            raise ValueError("model_path must be a nonempty string")
        model_path = Path(config["model_path"]).expanduser()
        config["model_path"] = str(model_path if model_path.is_absolute() else path.parent / model_path)
    return model_id, config


def image_part(part):
    """Validate the inline PNG/JPEG representation produced by read_pic.sh."""
    if set(part) != {"type", "image_url"} or not isinstance(part["image_url"], dict):
        raise RequestError("image_url must be an object containing url")
    image = part["image_url"]
    if set(image) - {"url", "detail"} or image.get("detail", "auto") != "auto":
        raise RequestError("Image detail supports only auto")
    url = image.get("url")
    if not isinstance(url, str):
        raise RequestError("image_url.url must be a data URL string")
    header, separator, encoded = url.partition(",")
    signatures = {"data:image/png;base64": b"\x89PNG\r\n\x1a\n",
                  "data:image/jpeg;base64": b"\xff\xd8\xff"}
    if not separator or header not in signatures:
        raise RequestError("Images must be inline data:image/png;base64 or data:image/jpeg;base64")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise RequestError(f"Image exceeds {MAX_IMAGE_BYTES} bytes")
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RequestError("Invalid image base64") from exc
    if len(decoded) > MAX_IMAGE_BYTES:
        raise RequestError(f"Image exceeds {MAX_IMAGE_BYTES} bytes")
    if not decoded.startswith(signatures[header]):
        raise RequestError("Image bytes do not match the declared PNG/JPEG type")
    return {"type": "image_url", "image_url": {"url": url}}


def normalize_content(content, role):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise RequestError("Message content must be text or an array of text/image_url blocks")
    parts, has_images = [], False
    for part in content:
        if not isinstance(part, dict):
            raise RequestError("Content blocks must be objects")
        if (part.get("type") == "text" and set(part) == {"type", "text"}
            and isinstance(part["text"], str)):
            parts.append(dict(part))
        elif part.get("type") == "image_url":
            if role != "user":
                raise RequestError("Images are supported only in user messages")
            parts.append(image_part(part))
            has_images = True
        else:
            raise RequestError("Supported content blocks: text and image_url (PNG/JPEG)")
    return parts if has_images else "".join(p["text"] for p in parts)


def has_images(messages):
    return any(isinstance(m["content"], list) and
               any(p["type"] == "image_url" for p in m["content"]) for m in messages)


def parse_request(body, model_id, config):
    if not isinstance(body, dict):
        raise RequestError("Request must be a JSON object")
    allowed = {
        "model", "messages", "stream", "stream_options", "max_tokens",
        "max_completion_tokens", "temperature", "top_p", "top_k", "seed",
        "reasoning_effort", "thinking", "n", "user",
    }
    unknown = set(body) - allowed
    if unknown:
        raise RequestError(f"Unsupported request fields: {sorted(unknown)}")
    if body.get("model") != model_id:
        raise RequestError(f"Server model is {model_id!r}")
    if type(body.get("stream", False)) is not bool:
        raise RequestError("stream must be boolean")
    stream_options = body.get("stream_options", {})
    if (not isinstance(stream_options, dict)
        or set(stream_options) - {"include_usage"}
        or type(stream_options.get("include_usage", False)) is not bool):
        raise RequestError("stream_options supports only boolean include_usage")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1:
        raise RequestError("Only n=1 is supported")
    if body.get("reasoning_effort", "none") != "none" or body.get("thinking", False) is not False:
        raise RequestError("Only reasoning_effort=none / thinking=false is supported")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a nonempty array")
    normalized = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise RequestError("Each message must have exactly role and content")
        role, content = message["role"], message["content"]
        if role not in ("system", "user", "assistant") or (role == "system" and i != 0):
            raise RequestError("Supported roles: initial system, user, assistant")
        content = normalize_content(content, role)
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "user":
        raise RequestError("Last message must have role=user")
    for key in ("max_tokens", "max_completion_tokens"):
        if key in body:
            positive_int(body[key], key)
    if ("max_tokens" in body and "max_completion_tokens" in body
        and body["max_tokens"] != body["max_completion_tokens"]):
        raise RequestError("Conflicting max_tokens and max_completion_tokens")
    limit = body.get("max_completion_tokens", body.get("max_tokens", config.get("max_output_tokens", 1024)))
    options = {
        "sampler": sampler_values({**config, **body}),
        "max_output_tokens": min(limit, config.get("max_num_tokens", 8192) - 1),
    }
    return normalized, options


@dataclass
class Slot:
    context: object
    history: list


class InferenceService:
    def __init__(self, runtime, model_id, config, max_contexts=2, max_vision_contexts=1):
        self.runtime, self.model_id, self.config = runtime, model_id, config
        self.max_contexts = positive_int(max_contexts, "max_contexts")
        self.slots = OrderedDict()
        self.vision_slots = OrderedDict()
        self.max_vision_contexts = positive_int(max_vision_contexts, "max_vision_contexts")
        self.lock = threading.Lock()

    def complete(self, body, emit=None):
        messages, options = parse_request(body, self.model_id, self.config)
        base = messages[:1] if messages[0]["role"] == "system" else []
        key = json.dumps([base, options], sort_keys=True, ensure_ascii=False)
        vision = has_images(messages)
        slots = self.vision_slots if vision else self.slots
        capacity = self.max_vision_contexts if vision else self.max_contexts
        # A single lock covers context selection, prefill, native async decode,
        # delivery and history commit. No two calls may use the engine together.
        with self.lock:
            started = time.monotonic()
            slot = slots.get(key)
            cold = slot is None
            mode = "cold"
            try:
                if cold:
                    if len(slots) >= capacity:
                        _, old = slots.popitem(last=False)
                        old.context.close()
                        LOG.info("CONTEXT_EVICT kind=%s slots=%d", "vision" if vision else "text", len(slots))
                    factory = self.runtime.new_vision_context if vision else self.runtime.new_context
                    slot = Slot(factory(base, options), deepcopy(base))
                    slots[key] = slot
                continuation = not cold and messages[:-1] == slot.history
                if vision:
                    mode = "vision-resume" if continuation else "vision-rebuild"
                elif not cold:
                    mode = "resume" if continuation else ("base" if messages[:-1] == base else "replay")
                LOG.info("CONTEXT mode=%s base_messages=%d history_messages=%d",
                         mode, len(base), len(messages) - 1)
                cached, evaluated = slot.context.prefill(messages, continuation)
                if cold:
                    evaluated += cached
                    cached = 0
                prompt_tokens = cached + evaluated
                if prompt_tokens >= self.config.get("max_num_tokens", 8192):
                    raise RequestError("Context is full; shorten messages or increase max_num_tokens")
                prefill_seconds = time.monotonic() - started
                first_token_seconds = None
                parts = []
                disconnected = False
                for text in slot.context.decode():
                    parts.append(text)
                    if text and first_token_seconds is None:
                        first_token_seconds = time.monotonic() - started
                    if emit and text and not disconnected:
                        try:
                            emit(text)
                        except OSError:
                            # Drain the native iterator before closing its C pointer.
                            disconnected = True
                completion_tokens = slot.context.finish()
                if vision:
                    cached = slot.context.cached_tokens
                    prompt_tokens = slot.context.prompt_tokens
                    evaluated = prompt_tokens - cached
                content = "".join(parts)
                if not content:
                    raise RuntimeError("LiteRT generated no text; context may be exhausted")
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": cached},
                }
                finish_reason = "length" if (
                    completion_tokens >= options["max_output_tokens"]
                    or usage["total_tokens"] >= self.config.get("max_num_tokens", 8192)
                ) else "stop"
                if disconnected:
                    raise ConnectionError("Client disconnected; native decode drained")
                if finish_reason == "length":
                    # A truncated raw decode may lack the model's end-of-turn
                    # marker. Replay explicit history from the base next time.
                    slot.history = []
                else:
                    slot.history = deepcopy(messages) + [{"role": "assistant", "content": content}]
                slots.move_to_end(key)
                LOG.info("REQUEST_DONE mode=%s cached=%d evaluated=%d decoded=%d "
                         "prefill_seconds=%s ttft_seconds=%s total_seconds=%.3f",
                         mode, cached, evaluated, completion_tokens,
                         "unknown" if vision else f"{prefill_seconds:.3f}",
                         f"{first_token_seconds:.3f}" if first_token_seconds is not None else "unknown",
                         time.monotonic() - started)
                return content, usage, finish_reason
            except BaseException:
                # Do not reuse partially written / failed native KV state.
                broken = slots.pop(key, None)
                if broken:
                    broken.context.close()
                raise

    def close(self):
        with self.lock:
            for slot in list(self.slots.values()) + list(self.vision_slots.values()):
                slot.context.close()
            self.slots.clear()
            self.vision_slots.clear()
            self.runtime.close()


class Server(ThreadingHTTPServer):
    # server_close waits for native generation in request threads before freeing
    # the shared engine. No abrupt deletion during a callback.
    daemon_threads = False

    def __init__(self, address, service):
        self.service = service
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        LOG.info("HTTP %s", fmt % args)

    def _json(self, status, value):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "invalid_request_error"
                                     if status < 500 else "server_error"}})

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ready", "model": self.server.service.model_id})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": self.server.service.model_id, "object": "model", "created": 0,
                "owned_by": "local",
            }]})
        else:
            self._error(404, "Unknown endpoint")

    def _sse(self, value):
        data = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    def do_POST(self):
        streaming_started = False
        try:
            if self.path != "/v1/chat/completions":
                self._error(404, "Unknown endpoint")
                return
            if self.headers.get("Transfer-Encoding"):
                raise RequestError("Transfer-Encoding is not supported; send Content-Length")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise RequestError("Invalid Content-Length") from exc
            if not 0 < length <= MAX_BODY_BYTES:
                self._error(413, f"Body must be between 1 and {MAX_BODY_BYTES} bytes")
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise RequestError("Incomplete request body")
            body = json.loads(raw)
            parse_request(body, self.server.service.model_id, self.server.service.config)
            response_id = "chatcmpl-" + uuid.uuid4().hex
            created = int(time.time())
            common = {"id": response_id, "created": created, "model": self.server.service.model_id}
            stream = body.get("stream", False)

            def emit(text):
                nonlocal streaming_started
                if not streaming_started:
                    # Mark before any write so a disconnected initial write
                    # cannot cause a second HTTP status line on this socket.
                    streaming_started = True
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self._sse({**common, "object": "chat.completion.chunk", "choices": [{
                        "index": 0, "delta": {"role": "assistant"}, "finish_reason": None,
                    }]})
                self._sse({**common, "object": "chat.completion.chunk", "choices": [{
                    "index": 0, "delta": {"content": text}, "finish_reason": None,
                }]})

            content, usage, reason = self.server.service.complete(body, emit if stream else None)
            if stream:
                # Include usage even without stream_options for cat-agent's
                # existing streaming client, which accepts it in the final chunk.
                self._sse({**common, "object": "chat.completion.chunk", "choices": [{
                    "index": 0, "delta": {}, "finish_reason": reason,
                }], "usage": usage})
                self._sse("[DONE]")
            else:
                self._json(200, {**common, "object": "chat.completion", "choices": [{
                    "index": 0, "message": {"role": "assistant", "content": content},
                    "finish_reason": reason,
                }], "usage": usage})
        except (OSError, ConnectionError):
            LOG.info("HTTP client disconnected or timed out")
        except Exception as exc:
            status = 400 if isinstance(exc, (RequestError, json.JSONDecodeError, UnicodeError)) else 500
            if status == 500:
                LOG.exception("Inference request failed")
            try:
                if streaming_started:
                    self._sse({"error": {"message": str(exc), "type": "server_error"}})
                    # No [DONE]: generation did not complete successfully.
                else:
                    self._error(status, str(exc))
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", help="Select a model when config lists more than one")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9379)
    parser.add_argument("--max-contexts", type=int, default=2,
                        help="Maximum resident system-prompt/settings slots (default: 2)")
    parser.add_argument("--max-vision-contexts", type=int, default=1,
                        help="Additional resident image conversations (default: 1)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    positive_int(args.max_contexts, "max-contexts")
    positive_int(args.max_vision_contexts, "max-vision-contexts")
    model_id, config = load_config(args.config, args.model)
    runtime = LiteRTRuntime(model_id, config)
    service = InferenceService(runtime, model_id, config, args.max_contexts, args.max_vision_contexts)
    try:
        with Server((args.host, args.port), service) as server:
            LOG.info("SERVER_READY http://%s:%d/v1 model=%s slots=%d",
                     args.host, args.port, model_id, args.max_contexts)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                LOG.info("Shutdown requested; waiting for active inference")
    finally:
        service.close()
