from __future__ import annotations

import re


_SENTENCE_END_RE = re.compile(
    r'(?:[!?…]+(?:["»”\)\]]+)?(?=\s|$)|\.(?:["»”\)\]]+)?(?=\s))'
)
_SOFT_BOUNDARY_RE = re.compile(r"[,;:—]")
_SHORT_LEAD_RE = re.compile(
    r"^(?:шаг|пункт|этап)\s+[^.!?]{1,24}[.!?]$",
    re.IGNORECASE,
)


class ManagerVoiceStream:
    """Expose only human-facing ASK/REPLY chunks from manager model events."""

    def __init__(self) -> None:
        self._mode = "idle"
        self._buffer = ""

    def decode_start(self) -> None:
        self._mode = "pending"
        self._buffer = ""

    def feed_chunk(self, chunk: str) -> str:
        if not chunk or self._mode in {"idle", "hidden"}:
            return ""
        if self._mode == "visible":
            return chunk

        self._buffer += chunk
        text = self._buffer
        candidates = ("/work#", "ASK\n", "REPLY\n")

        if text.startswith("/work#"):
            self._mode = "hidden"
            self._buffer = ""
            return ""

        for prefix in ("ASK\n", "REPLY\n"):
            if text.startswith(prefix):
                self._mode = "visible"
                self._buffer = ""
                return text[len(prefix) :]

        if any(candidate.startswith(text) for candidate in candidates):
            return ""

        if "\n" in text:
            self._mode = "hidden"
            self._buffer = ""

        return ""


class TextFragmenter:
    """Adaptive streaming text fragmenter retained from the proven voice path."""

    def __init__(
        self,
        first_max_chars: int = 90,
        target_chars: int = 180,
        max_chars: int = 260,
        target_sentences: int = 3,
        min_soft_chars: int = 45,
    ) -> None:
        self._buffer = ""
        self._pending: list[str] = []
        self._short_lead: str | None = None
        self._first_emitted = False
        self._first_max_chars = first_max_chars
        self._target_chars = target_chars
        self._max_chars = max_chars
        self._target_sentences = target_sentences
        self._min_soft_chars = min_soft_chars

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        return self._extract(final=False)

    def finish(self) -> list[str]:
        return self._extract(final=True)

    def _extract(self, final: bool) -> list[str]:
        fragments: list[str] = []

        while self._buffer:
            sentence_match = _SENTENCE_END_RE.search(self._buffer)
            if sentence_match is not None:
                cut = sentence_match.end()
                sentence = self._buffer[:cut].strip()
                self._buffer = self._buffer[cut:].lstrip()
                if sentence:
                    self._accept_sentence(sentence, fragments)
                continue

            limit = self._first_max_chars if not self._first_emitted else self._max_chars
            if len(self._buffer) >= limit:
                cut = self._find_soft_cut(limit)
                piece = self._buffer[:cut].strip()
                self._buffer = self._buffer[cut:].lstrip()
                if piece:
                    self._accept_piece(piece, fragments)
                continue

            break

        if final:
            tail = self._buffer.strip()
            self._buffer = ""
            if tail:
                self._accept_sentence(tail, fragments)

            if self._short_lead is not None:
                self._accept_piece(self._short_lead, fragments)
                self._short_lead = None

            self._flush_pending(fragments)

        return fragments

    def _accept_sentence(self, sentence: str, fragments: list[str]) -> None:
        if self._short_lead is not None:
            sentence = f"{self._short_lead} {sentence}"
            self._short_lead = None
        elif _SHORT_LEAD_RE.fullmatch(sentence):
            self._short_lead = sentence
            return

        self._accept_piece(sentence, fragments)

    def _accept_piece(self, piece: str, fragments: list[str]) -> None:
        if not self._first_emitted:
            fragments.append(piece)
            self._first_emitted = True
            return

        self._pending.append(piece)
        pending_text = " ".join(self._pending)

        if (
            len(pending_text) >= self._target_chars
            or len(self._pending) >= self._target_sentences
        ):
            self._flush_pending(fragments)

    def _flush_pending(self, fragments: list[str]) -> None:
        if not self._pending:
            return

        fragment = " ".join(self._pending).strip()
        self._pending.clear()
        if fragment:
            fragments.append(fragment)

    def _find_soft_cut(self, limit: int) -> int:
        window = self._buffer[:limit]

        soft_cut = -1
        for match in _SOFT_BOUNDARY_RE.finditer(window):
            if match.end() >= self._min_soft_chars:
                soft_cut = match.end()

        if soft_cut > 0:
            return soft_cut

        space_cut = window.rfind(" ", self._min_soft_chars)
        if space_cut > 0:
            return space_cut

        return limit
