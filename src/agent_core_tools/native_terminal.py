"""Opt-in native CORE TUI: reuse the existing curses interface, not the old CORE.

Uses the native CORE IPC contract directly. Durable TASK QUERY notices are
fetched on connection, displayed once per terminal session and ACKed via
the authoritative Task SYSTEM path through the native CORE.
"""
from __future__ import annotations

import argparse
from collections import deque
import logging
from pathlib import Path
import queue
import sys
import time

from litert_agent.terminal import CoreClient, TerminalTUI
from .server import DEFAULT_SOCKET

LOGGER = logging.getLogger(__name__)
INBOX_POLL_SECONDS = 5.0
ACK_RETRY_SECONDS = 2.0
MAX_SESSION_QUERY_IDS = 4096


class NativeTerminalTUI(TerminalTUI):
    """Native IPC adaptation; no modification of legacy TUI or CORE."""

    def __init__(self, client: CoreClient, core_info: dict, status: dict):
        super().__init__(client, core_info, status)
        self._query_seen: set[str] = set()
        self._query_acked: set[str] = set()
        self._query_order: deque[str] = deque()
        self._pending_query_ack: dict[str, float] = {}
        self._next_inbox_poll = 0.0

    def _request_snapshot(self) -> None:
        super()._request_snapshot()
        now = time.monotonic()
        if now >= self._next_inbox_poll:
            # An unsuccessful request does NOT imply an empty inbox.
            self._next_inbox_poll = now + INBOX_POLL_SECONDS
            try:
                self.client.send({"type": "task_query_results"})
            except OSError as exc:
                LOGGER.warning("Native QUERY inbox poll failed: %s", exc)
        for run_id, next_attempt in tuple(self._pending_query_ack.items()):
            if now < next_attempt:
                continue
            self._pending_query_ack[run_id] = now + ACK_RETRY_SECONDS
            try:
                self.client.send({"type": "task_query_ack", "run_id": run_id})
            except OSError as exc:
                LOGGER.warning("Native QUERY ACK pending run=%s: %s", run_id, exc)

    def _receive_query(self, item: dict) -> None:
        run_id, text = item.get("run_id"), item.get("text")
        if not isinstance(run_id, str) or not run_id or not isinstance(text, str):
            LOGGER.warning("Malformed native QUERY notice ignored")
            return
        if run_id not in self._query_seen:
            self._append_dialog("TASK QUERY", text)
            self._query_seen.add(run_id)
            self._query_order.append(run_id)
            # Keep session memory bounded. Unacknowledged RUNs are not evicted
            # from the display deduplication set.
            while len(self._query_order) > MAX_SESSION_QUERY_IDS:
                oldest = self._query_order.popleft()
                if oldest in self._pending_query_ack:
                    self._query_order.append(oldest)
                    break
                self._query_seen.discard(oldest)
                self._query_acked.discard(oldest)
        # Queue ACK only after the text has been placed in terminal history.
        # An absent ACK response is retried; the TASK SYSTEM ACK is idempotent.
        if run_id not in self._query_acked and run_id not in self._pending_query_ack:
            self._pending_query_ack[run_id] = 0.0

    def _draw_info(self, win) -> None:
        """Native CORE metrics only; avoid misleading legacy status fields."""
        height, width = win.getmaxyx()
        row = 1

        def put(key: str, value: object = "") -> None:
            nonlocal row
            if row >= height - 1:
                return
            self._safe_addstr(
                win, row, 2,
                f"{key:<12} {value}" if value != "" else key,
                0, max(0, width - 4),
            )
            row += 1

        put("NATIVE CORE")
        put("model", self.core_info.get("model") or "?")
        put("backend", self.core_info.get("backend") or "?")
        put("stage", self.core_info.get("stage", "?"))
        put("base KV", self.core_info.get("manager_warm_tokens", "?"))
        put("chat", "open" if self.status.get("chat_open") else "closed")
        row += 1

        put("SCHEDULER")
        put("state", self.status.get("state", "UNKNOWN"))
        put("active", self.status.get("label", "") or "-")
        put("queued", self.status.get("pending", "?"))
        priorities = self.status.get("queued_by_priority")
        if isinstance(priorities, dict):
            for kind in ("human", "hardware", "background", "control"):
                put(kind, priorities.get(kind, 0))
        row += 1

        worker = self.status.get("task_worker")
        if isinstance(worker, dict):
            put("TASK SYSTEM")
            put("ready", worker.get("ready", False))
            put("offered", worker.get("offered", 0))
            put("running", len(worker.get("reserved_agents", [])))
            put("finishes", worker.get("unacknowledged", 0))
            put("query RPC", worker.get("pending_query_validations", 0))
            put("inbox ACK", len(self._pending_query_ack))
            row += 1

        agents = self.status.get("agents")
        if isinstance(agents, list) and agents:
            put("AGENTS")
            for agent in agents:
                if isinstance(agent, dict):
                    put(str(agent.get("agent_id", "agent")),
                        agent.get("state", "?"))

    def _poll_core(self) -> None:
        while True:
            try:
                item = self.client.incoming.get_nowait()
            except queue.Empty:
                return
            if not isinstance(item, dict):
                continue
            kind = str(item.get("type", ""))
            if kind == "task_query_result":
                self._receive_query(item)
            elif kind == "task_query_results":
                items = item.get("items")
                if isinstance(items, list):
                    for notice in items:
                        if isinstance(notice, dict):
                            self._receive_query(notice)
                else:
                    self._append_dialog("CORE", "Некорректный ответ QUERY inbox")
            elif kind == "task_query_acked":
                run_id = item.get("run_id")
                if isinstance(run_id, str):
                    if item.get("acknowledged"):
                        self._query_acked.add(run_id)
                    else:
                        LOGGER.warning("Native QUERY ACK no longer found run=%s", run_id)
                    self._pending_query_ack.pop(run_id, None)
            elif kind == "snapshot":
                snapshot = item.get("status")
                if isinstance(snapshot, dict):
                    self.status = {"chat_open": self.status.get("chat_open", False),
                                   **snapshot}
            elif kind == "status":
                self.status = {**self.status, **item}
            elif kind == "model_event":
                self._handle_model_event(item)
            elif kind == "reply":
                self._finalize_manager_text(str(item.get("text", "")))
            elif kind == "completed":
                # New CORE does not stream old model_event chunks, but its
                # final completion still carries the dialogue-open flag.
                if isinstance(item.get("chat_open"), bool):
                    self.status["chat_open"] = item["chat_open"]
            elif kind == "notification":
                # A legacy notification is not a TASK QUERY result and must
                # not be ACKed via task_query_ack.
                self._append_dialog("CORE", str(item.get("text", "")))
            elif kind == "busy":
                self._append_dialog("CORE", str(item.get("text", "Гена занят")))
            elif kind == "error":
                # Inference errors carry their explanation in "text"; protocol
                # validation failures instead use "error".
                self._append_dialog("CORE", str(
                    item.get("error") or item.get("text") or "CORE error",
                ))
            elif kind == "disconnected":
                self._append_dialog("CORE", str(item.get("text", "CORE disconnected")))
                self._quit = True
            # Native 'accepted', 'completed', 'released' and 'pong' require no
            # additional terminal rendering; the final 'reply' is authoritative.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native CORE terminal (opt-in)")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET,
                        help="Native CORE Unix socket; no Task SYSTEM direct access")
    args = parser.parse_args(argv)
    client = CoreClient(args.socket)
    try:
        try:
            first = client.connect()
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"Cannot connect to native CORE at {args.socket}: {exc}",
                  file=sys.stderr)
            return 2
        if first.get("type") == "busy":
            print(f"Гена занят ({first.get('owner') or 'unknown'})")
            return 3
        if first.get("type") != "acquired":
            print(f"Unexpected native CORE response: {first}", file=sys.stderr)
            return 2
        core_info = first.get("core")
        status = first.get("status")
        NativeTerminalTUI(
            client,
            core_info if isinstance(core_info, dict) else {},
            status if isinstance(status, dict) else {},
        ).run()
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
