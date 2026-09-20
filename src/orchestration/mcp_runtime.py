"""Optional official-SDK client pool with a frozen startup tool catalog.

Each connection is entered, used and exited by one owner task. This is important
for the SDK's AnyIO cancel scopes. No tools/call is retried after dispatch.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future, InvalidStateError, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os
import re
import threading
from types import MappingProxyType

from .mcp_config import McpServerConfig
from .skills import Skill
from .mcp_result import format_mcp_result

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def sdk_connection(config):
    # Optional imports must remain inside the enabled MCP path.
    from mcp import Client, StdioServerParameters
    if config.transport == "stdio":
        env = {key: os.environ[source] for key, source in config.env}
        target = StdioServerParameters(command=config.command, args=list(config.args), env=env)
        async with Client(target, cache=None) as client:
            yield client
    else:
        import httpx2
        from mcp.client.streamable_http import streamable_http_client
        headers = {key: os.environ[source] for key, source in config.headers_env}
        async with httpx2.AsyncClient(headers=headers, timeout=config.call_timeout_seconds) as http:
            transport = streamable_http_client(config.url, http_client=http)
            async with Client(transport, cache=None) as client:
                yield client


@dataclass(frozen=True)
class McpTool:
    server: str
    name: str
    description: str
    schema_json: str

    @property
    def qualified_name(self):
        return f"mcp:{self.server}:{self.name}"

    def skill(self):
        definition = {"name": self.qualified_name, "description": self.description,
                      "inputSchema": json.loads(self.schema_json)}
        return Skill(self.qualified_name, self.description,
                     "Call /work#" + self.qualified_name + " {JSON}; wait for runtime result.\n"
                     + json.dumps(definition, ensure_ascii=False, sort_keys=True))


@dataclass
class _Request:
    tool: McpTool
    arguments: dict
    result: Future
    started: bool = False


class McpRuntime:
    def __init__(self, configs, *, connection_factory=sdk_connection):
        self.configs = tuple(c for c in configs if c.enabled)
        self._factory = connection_factory
        self._tools = MappingProxyType({})
        self._status = {c.name: ("unavailable", "not_started") for c in self.configs}
        self._lock = threading.Lock()
        self._started = threading.Event()
        self._thread = None
        self._loop = None
        self._stop = None
        self._queues = {}
        self._closed = False

    def _state(self, name, state, reason=""):
        with self._lock:
            previous = self._status.get(name)
            self._status[name] = (state, reason)
        if previous != (state, reason):
            # No URLs, command arguments, headers or exception strings (secrets).
            LOGGER.info("MCP server=%s state=%s reason=%s", name, state, reason or "none")

    def snapshot(self):
        with self._lock:
            return {name: {"state": value[0], "reason": value[1]}
                    for name, value in self._status.items()}

    def start(self):
        if self._thread is not None or self._closed or not self.configs:
            return
        self._thread = threading.Thread(target=self._run, name="cat-agent-mcp", daemon=True)
        self._thread.start()
        # Owners each bound their first attempt; discovery runs concurrently.
        timeout = max(c.connect_timeout_seconds for c in self.configs) + 10
        if not self._started.wait(timeout):
            self.close()
            for config in self.configs:
                self._state(config.name, "unavailable", "startup_timeout")

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as exc:
            for config in self.configs:
                self._state(config.name, "unavailable", type(exc).__name__)
        finally:
            self._started.set()

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        self._queues = {c.name: asyncio.Queue() for c in self.configs}
        initial = {c.name: asyncio.Future() for c in self.configs}
        owners = [asyncio.create_task(self._owner(c, initial[c.name])) for c in self.configs]
        try:
            results = await asyncio.gather(*initial.values())
            tools = {}
            for group in results:
                tools.update({t.qualified_name: t for t in group})
            self._tools = MappingProxyType(tools)
            LOGGER.info("MCP catalog frozen: servers=%d tools=%d", len(self.configs), len(tools))
            self._started.set()
            await self._stop.wait()
        finally:
            for task in owners:
                task.cancel()
            await asyncio.gather(*owners, return_exceptions=True)
            for queue in self._queues.values():
                while not queue.empty():
                    self._finish(queue.get_nowait(), "SYSTEM_ERROR\nMCP unavailable: CORE stopping")

    @staticmethod
    async def _list_tools(client, server):
        if client.server_capabilities.tools is None:
            return ()
        tools = {}
        cursor = None
        cursors = set()
        while True:
            page = await client.list_tools(cursor=cursor)
            for item in page.tools:
                # Restrict only the wire-facing identifier, never rewrite names.
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", item.name) or item.name in tools:
                    raise ValueError("invalid_or_duplicate_tool_name")
                if not isinstance(item.input_schema, dict):
                    raise ValueError("invalid_tool_schema")
                tools[item.name] = McpTool(server, item.name, item.description or "",
                    json.dumps(item.input_schema, ensure_ascii=False, sort_keys=True, allow_nan=False))
            cursor = page.next_cursor
            if cursor is None:
                return tuple(tools[name] for name in sorted(tools))
            if cursor in cursors:
                raise ValueError("repeated_tools_list_cursor")
            cursors.add(cursor)

    async def _owner(self, config, initial):
        try:
            import anyio
        except ImportError:
            self._state(config.name, "unavailable", "sdk_missing_install_mcp_extra")
            initial.set_result(())
            return
        current = None
        try:
            while True:
                try:
                    # Keep this scope OUTSIDE the entire SDK session lifetime.
                    with anyio.fail_after(config.connect_timeout_seconds) as startup_scope:
                        async with self._factory(config) as client:
                            discovered = await self._list_tools(client, config.name)
                            live = {t.name: t for t in discovered}
                            if not initial.done():
                                initial.set_result(discovered)
                            startup_scope.deadline = float("inf")
                            self._state(config.name, "ready")
                            while True:
                                current = await self._queues[config.name].get()
                                if current.result.cancelled():
                                    current = None
                                    continue
                                # Reconnect can verify current definitions but never changes BASE.
                                now = live.get(current.tool.name)
                                if now is None or now.schema_json != current.tool.schema_json:
                                    self._finish(current, "SYSTEM_ERROR\nMCP tool_changed: restart CORE to refresh catalog")
                                    current = None
                                    continue
                                # Atomically commit dispatch versus a caller timing out in the queue.
                                if not current.result.set_running_or_notify_cancel():
                                    current = None
                                    continue
                                current.started = True
                                try:
                                    with anyio.fail_after(config.call_timeout_seconds):
                                        # Low-level official API avoids automatic input-required rounds.
                                        result = await client.session.call_tool(
                                            current.tool.name, current.arguments,
                                            read_timeout_seconds=config.call_timeout_seconds)
                                    self._finish(current, format_mcp_result(result))
                                    current = None
                                except Exception as exc:
                                    # Publish unavailability BEFORE releasing the caller and
                                    # before SDK __aexit__, which can wait for a slow child.
                                    self._state(config.name, "unavailable", type(exc).__name__)
                                    self._finish(current, "SYSTEM_ERROR\nMCP outcome_unknown: call failed or timed out; do not replay automatically")
                                    current = None
                                    raise
                except Exception as exc:
                    reason = "sdk_missing_install_mcp_extra" if isinstance(exc, ImportError) else type(exc).__name__
                    self._state(config.name, "unavailable", reason)
                    if not initial.done():
                        initial.set_result(())
                    if isinstance(exc, ImportError):
                        return
                    await asyncio.sleep(config.reconnect_delay_seconds)
        finally:
            if not initial.done():
                initial.set_result(())
            if current is not None:
                self._finish(current, "SYSTEM_ERROR\nMCP outcome_unknown: CORE stopping; do not replay automatically"
                             if current.started else "SYSTEM_ERROR\nMCP unavailable: CORE stopping")
            if self._closed:
                self._state(config.name, "unavailable", "closed")
            elif self.snapshot()[config.name]["state"] == "ready":
                self._state(config.name, "unavailable", "connection_owner_stopped")

    @staticmethod
    def _finish(request, text):
        try:
            request.result.set_result(text)
        except InvalidStateError:
            pass  # Caller may have timed out/cancelled while the owner was finishing.

    def skills(self):
        return tuple(tool.skill() for tool in self._tools.values())

    def call(self, name, arguments):
        tool = self._tools.get(name)
        if tool is None:
            return "SYSTEM_ERROR\nMCP unknown_tool: not in CORE startup catalog"
        if self._closed or self._loop is None or not self._thread.is_alive():
            return "SYSTEM_ERROR\nMCP unavailable: CORE connection closed"
        if self.snapshot()[tool.server]["state"] != "ready":
            return "SYSTEM_ERROR\nMCP unavailable: server reconnecting"
        try:
            from jsonschema.validators import validator_for
            from referencing import Registry
            schema = json.loads(tool.schema_json)
            validator = validator_for(schema)
            validator.check_schema(schema)
            # No implicit retrieval of remote $ref documents.
            validator(schema, registry=Registry()).validate(arguments)
        except Exception:
            return "SYSTEM_ERROR\nMCP invalid_arguments: arguments do not match startup inputSchema"
        request = _Request(tool, arguments, Future())
        config = next(c for c in self.configs if c.name == tool.server)
        try:
            self._loop.call_soon_threadsafe(self._queues[tool.server].put_nowait, request)
            return request.result.result(timeout=config.call_timeout_seconds + 1)
        except FutureTimeout:
            if request.result.cancel():
                return "SYSTEM_ERROR\nMCP timeout: expired before dispatch; no call sent"
            return "SYSTEM_ERROR\nMCP outcome_unknown: call timed out; do not replay automatically"
        except RuntimeError:
            return "SYSTEM_ERROR\nMCP unavailable: connection loop stopped"

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._stop is not None:
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                LOGGER.error("MCP connection cleanup exceeded shutdown timeout")
