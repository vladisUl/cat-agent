"""Bound the stock SDK automatic loop through its public tool event hooks.

Hook exceptions propagate outside the SDK's tool.execute exception handler.
No native ABI patch and no replay of a partially completed operation.
"""
from __future__ import annotations

import inspect
from contextvars import ContextVar
import json
import math
from pathlib import Path
import time
from typing import get_type_hints
import uuid

TURN_ACTIVE = ContextVar('native_turn_active', default=None)
TURN_BUDGET = ContextVar('native_turn_budget', default=None)


class ToolTurnStopped(RuntimeError):
    """The automatic loop must stop, rather than feeding another tool error."""


def typed_signature(function):
    original = inspect.unwrap(function)
    hints = get_type_hints(original)
    signature = inspect.signature(original)
    return signature.replace(
        parameters=[p.replace(annotation=hints.get(name, p.annotation))
                    for name, p in signature.parameters.items()],
        return_annotation=hints.get('return', signature.return_annotation),
    )


def wire_size(value):
    # Count escaped JSON bytes, not characters. This deliberately overestimates
    # ordinary text tokens; it is NOT a model tokenizer or an exact KV estimate.
    return len(json.dumps(value, ensure_ascii=True, separators=(',', ':')).encode())


class ResultStore:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def save(self, value):
        self.directory.mkdir(parents=True, exist_ok=True)
        identifier = uuid.uuid4().hex
        path = self.directory / (identifier + '.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False)
        return identifier

    def tool(self):
        def read_tool_result(result_id: str, offset: int = 0, length: int = 128) -> dict:
            """Прочитать сохранённый большой результат инструмента порциями.

            Args:
                result_id (str): Идентификатор из сокращённого результата.
                offset (int): Смещение в символах, начиная с нуля.
                length (int): Число символов от 1 до 512.
            """
            if (not isinstance(result_id, str) or len(result_id) != 32
                    or any(c not in '0123456789abcdef' for c in result_id)):
                return {'ok': False, 'error': 'invalid_result_id'}
            if type(offset) is not int or offset < 0 or type(length) is not int or not 1 <= length <= 512:
                return {'ok': False, 'error': 'invalid_page'}
            try:
                with (self.directory / (result_id + '.json')).open(encoding='utf-8') as stream:
                    # Bounded reads, including when skipping a large prefix.
                    remaining = offset
                    while remaining:
                        part = stream.read(min(remaining, 4096))
                        if not part:
                            return {'ok': False, 'error': 'offset_past_end'}
                        remaining -= len(part)
                    text = stream.read(length)
                    more = bool(stream.read(1))
            except OSError as exc:
                return {'ok': False, 'error': str(exc)}
            return {'ok': True, 'text': text,
                    'next_offset': offset + len(text) if more else None}
        return read_tool_result


class ToolGuard:
    def __init__(self, ledger, *, max_calls=12, timeout=180.0,
                 max_context_tokens=4096, max_output_tokens=512,
                 max_result_bytes=2048, store=None):
        if (max_calls < 1 or not math.isfinite(timeout) or timeout <= 0 or max_result_bytes < 256
                or max_output_tokens < 1 or max_context_tokens <= max_output_tokens + 256):
            raise ValueError('Invalid native tool/context limits')
        self.ledger = ledger
        self.max_calls = max_calls
        self.timeout = timeout
        self.context_limit = max_context_tokens
        self.output_reserve = max_output_tokens
        self.max_result_bytes = max_result_bytes
        self.store = store
        self.conversation = None
        self.reset()

    def reset(self, is_active=None):
        self.attempts = 0
        self.deadline = time.monotonic() + self.timeout
        self.shared = TURN_BUDGET.get()
        if self.shared is not None:
            self.shared.setdefault('deadline', self.deadline)
            self.shared.setdefault('calls', 0)
            self.deadline = self.shared['deadline']
        self.is_active = is_active or TURN_ACTIVE.get() or (lambda: True)
        self.batch_kv = None
        self.batch_bytes = 0

    def _stop(self, reason):
        self.ledger.status_code = 1
        raise ToolTurnStopped(reason)

    def _available(self):
        kv = self.conversation.token_count
        if kv != self.batch_kv:
            self.batch_kv, self.batch_bytes = kv, 0
        return self.context_limit - kv - self.output_reserve - 256 - self.batch_bytes

    def approve_tool_call(self, tool_call):
        if not self.is_active():
            self._stop('Native turn cancelled before next tool call')
        if time.monotonic() >= self.deadline:
            self._stop('Native tool turn deadline reached')
        calls = self.shared['calls'] if self.shared is not None else self.attempts
        if calls >= self.max_calls:
            self._stop('Native tool call limit reached')
        if self._available() < 256:
            self._stop('Insufficient context for tool result; use /new')
        self.attempts += 1
        if self.shared is not None:
            self.shared['calls'] += 1
        self.calls_before = self.ledger.calls
        return True

    def process_tool_response(self, result):
        # Unknown names and failures before callback entry bypass ledger.wrap.
        if self.ledger.calls == self.calls_before:
            self.ledger.calls += 1
        if (isinstance(result, str) and result.startswith('Error:')) or (
            isinstance(result, dict) and (result.get('ok') is False or result.get('isError') is True)
        ):
            self.ledger.status_code = 1
        budget = min(self.max_result_bytes, self._available() - 128)
        if wire_size(result) > budget:
            identifier = self.store.save(result) if self.store is not None else None
            summary = {'truncated': True, 'original_bytes': wire_size(result)}
            if isinstance(result, dict):
                for key in ('ok', 'isError', 'exit_code'):
                    if key in result:
                        summary[key] = result[key]
            if identifier is not None:
                summary['result_id'] = identifier
                summary['read_with'] = 'read_tool_result'
            else:
                summary['full_result_saved'] = False
            preview = json.dumps(result, ensure_ascii=False)
            summary['preview'] = ''
            low, high = 0, min(len(preview), max(0, budget))
            while low < high:
                mid = (low + high + 1) // 2
                summary['preview'] = preview[:mid]
                if wire_size(summary) <= budget:
                    low = mid
                else:
                    high = mid - 1
            summary['preview'] = preview[:low]
            result = summary
        if wire_size(result) > budget:
            self._stop('Insufficient context for tool result metadata; operation was not replayed')
        self.batch_bytes += wire_size(result) + 128
        if not self.is_active():
            self._stop('Native turn cancelled after tool execution; operation was not replayed')
        if time.monotonic() >= self.deadline:
            self._stop('Native tool turn deadline reached after execution; operation was not replayed')
        return result
