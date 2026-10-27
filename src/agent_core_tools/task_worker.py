"""Native CORE -> authoritative Task SYSTEM RPC worker (stage 5A).

The shared service owns schedules, MQTT and durable RUN state. This adapter
only heartbeats, claims offered TASKs, and executes authorized single-turn
native agents on the ONE serialized model scheduler. Never retries an action.
"""
from __future__ import annotations

from collections import OrderedDict
import logging
import threading
import time
import uuid

from task_system.client import TaskRPC
from orchestration.tasks import TaskStoreError
from .priority_scheduler import QueueFullError

LOGGER = logging.getLogger(__name__)
POLL_SECONDS = 0.5
HEARTBEAT_SECONDS = 5.0
MAX_UNACKNOWLEDGED = 32


class NativeTaskWorker:
    """No local task timers, database, event hooks or model-owned RPC callbacks."""

    def __init__(
        self, scheduler, manager, *, rpc=None, agent_permissions=None,
        agent_blocked=None, stop_event=None, poll_seconds=POLL_SECONDS,
        clock=time.monotonic,
    ):
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.scheduler = scheduler
        self.manager = manager
        self.rpc = rpc or TaskRPC()
        self.owner = uuid.uuid4().hex
        self.permissions = {
            name: frozenset(tools) for name, tools in (agent_permissions or {}).items()
        }
        if not self.permissions:
            raise ValueError("Task worker requires at least one configured agent")
        self.agent_blocked = agent_blocked or (lambda ident: False)
        self.external_stop = stop_event
        self.clock = clock
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread = None
        self._next_heartbeat = 0.0
        self._seen: set[str] = set()
        self._reserved: set[str] = set()
        self._pending_finishes: OrderedDict[str, tuple[dict, str]] = OrderedDict()
        self._lock = threading.RLock()
        self._ready = False

    def _rpc(self, method: str, **kwargs):
        return self.rpc.call("worker", method, owner=self.owner, **kwargs)

    def _can_run(self) -> bool:
        return not self._stop.is_set() and not (
            self.external_stop is not None and self.external_stop.is_set()
        )

    def start(self) -> None:
        if self._thread is not None:
            return
        # Fail fast: an unconnected Task SYSTEM must not look like READY.
        self._rpc("heartbeat", executor="litert", ready=True)
        self._ready = True
        self._next_heartbeat = self.clock() + HEARTBEAT_SECONDS
        self._thread = threading.Thread(target=self._loop, name="native-task-rpc", daemon=True)
        self._thread.start()
        LOGGER.info("native-task WORKER READY owner=%s", self.owner)

    def _heartbeat(self) -> bool:
        if self.clock() < self._next_heartbeat and self._ready:
            return True
        try:
            self._rpc("heartbeat", executor="litert", ready=True)
        except TaskStoreError as exc:
            LOGGER.warning("native-task HEARTBEAT failed: %s", exc)
            self._ready = False
            return False
        self._ready = True
        self._next_heartbeat = self.clock() + HEARTBEAT_SECONDS
        return True

    def _finish_pending(self) -> bool:
        with self._lock:
            entries = tuple(self._pending_finishes.items())
        for token, (event, outcome) in entries:
            try:
                acknowledged = self._rpc(
                    "finish", run_id=event["run_id"],
                    offer_id=event["offer_id"], outcome=outcome,
                )
            except TaskStoreError as exc:
                LOGGER.warning("native-task FINISH pending id=%s: %s", token, exc)
                return False
            if acknowledged is False:
                LOGGER.warning("native-task FINISH revoked token=%s (no replay)", token)
            with self._lock:
                self._pending_finishes.pop(token, None)
                self._seen.discard(token)
        return True

    def _select_agent(self, task: dict) -> str | None:
        required = task.get("skills")
        if not isinstance(required, list) or not all(isinstance(s, str) for s in required):
            return None
        agents = {
            s["agent_id"]: s["state"] for s in self.manager.agent_status()
        }
        # Never take the server lock while holding the TaskWorker lock:
        # CoreServer admission checks do the reverse (server -> worker).
        for ident, granted in self.permissions.items():
            if set(required) > granted or agents.get(ident) != "FREE":
                continue
            if self.agent_blocked(ident):
                continue
            with self._lock:
                if ident not in self._reserved:
                    return ident
        return None

    def poll_once(self) -> bool:
        """One non-blocking Task SYSTEM admission cycle (no model inference)."""
        # During CORE shutdown, keep the authoritative worker lease alive
        # until the model scheduler has drained its current (non-preemptible)
        # native inference. Only stop admitting new work.
        if not self._heartbeat():
            return False
        if not self._can_run():
            return False
        if not self._finish_pending():
            return False  # Never accumulate more work behind unknown finish replies.
        with self._lock:
            if len(self._seen) >= MAX_UNACKNOWLEDGED:
                return False
            seen = tuple(self._seen)
        try:
            event = self._rpc("pull", executor="litert", seen=list(seen))
        except TaskStoreError as exc:
            LOGGER.warning("native-task PULL unavailable: %s", exc)
            return False
        if event is None:
            return False
        if not isinstance(event, dict) or not event.get("run_id") or not event.get("offer_id"):
            LOGGER.error("native-task invalid offer (no RUN/offer ID)")
            return False
        token = event["offer_id"]
        if token in seen:
            return False
        try:
            task = self.rpc.call("tasks", "get", event["task_id"])
        except TaskStoreError as exc:
            LOGGER.warning("native-task TASK read failed: %s", exc)
            return False
        if (not isinstance(task, dict) or not task.get("enabled")
                or task.get("generation") != event.get("task_generation")):
            LOGGER.info("native-task stale offer id=%s (not executed)", event["run_id"])
            return False
        # 'query' requires result routing in the manager, implemented separately.
        # Keep the offer untouched rather than silently converting its semantics.
        if task.get("method") != "task":
            LOGGER.debug("native-task unsupported method %r id=%s", task.get("method"), event["run_id"])
            return False
        agent_id = self._select_agent(task)
        if agent_id is None:
            return False  # Keep the offer; permissions/state may change.
        source = event.get("source", "")
        kind = "hardware" if source == "mqtt" else "background"
        with self._lock:
            if agent_id in self._reserved or token in self._seen:
                return False
            # Reserve BEFORE publishing to the scheduler. A fast worker
            # can execute and finish before submit() returns.
            self._reserved.add(agent_id)
            self._seen.add(token)
            try:
                self.scheduler.submit(
                    kind, lambda: self._execute(event, task, agent_id),
                )
            except (QueueFullError, RuntimeError):
                self._reserved.discard(agent_id)
                self._seen.discard(token)
                return False  # Unadmitted offer is not marked seen.
        LOGGER.info("native-task ADMITTED run=%s offer=%s agent=%s priority=%s",
                    event["run_id"], token, agent_id, kind)
        return True

    def _execute(self, event: dict, task: dict, agent_id: str) -> None:
        """Runs on THE ONLY native model worker; no nested inference."""
        token = event["offer_id"]
        outcome = "cancelled"
        started = False
        try:
            if not self._can_run():
                return  # Offered but never begun: service can re-offer on offline.
            # Check agent again at the actual boundary, not just at admission.
            if self._select_agent_for_reserved(agent_id) is False:
                LOGGER.info("native-task AGENT BUSY run=%s; not beginning", event["run_id"])
                return
            try:
                allowed = self._rpc(
                    "begin", run_id=event["run_id"], offer_id=token,
                )
            except TaskStoreError:
                LOGGER.warning("native-task BEGIN unavailable run=%s; not executing", event["run_id"])
                return
            if not allowed:
                LOGGER.info("native-task REVOKED before start run=%s", event["run_id"])
                return
            started = True
            try:
                valid = self._rpc(
                    "valid", run_id=event["run_id"], offer_id=token,
                )
            except TaskStoreError:
                valid = False  # Fail closed: NEVER act without authoritative validity.
            if not valid:
                LOGGER.info("native-task revoked/unverifiable before model run=%s",
                            event["run_id"])
                return
            instruction = task["text"]
            # The prompt is taken from the saved TASK record, not the IPC offer.
            result = self.manager.run_agent(agent_id, instruction)
            from orchestration_tools.agents import AgentState
            outcome = "done" if result.state is AgentState.FREE else "error"
            LOGGER.info("native-task EXECUTED run=%s agent=%s outcome=%s",
                        event["run_id"], agent_id, outcome)
        except Exception:
            LOGGER.exception("native-task FAILED run=%s (never automatically replay)",
                             event["run_id"])
            outcome = "error"
        finally:
            with self._lock:
                self._reserved.discard(agent_id)
                if started:
                    # Finish is retried by the poll thread, NEVER the model call.
                    self._pending_finishes[token] = (event, outcome)
                else:
                    self._seen.discard(token)

    def _select_agent_for_reserved(self, agent_id: str) -> bool:
        state = {s["agent_id"]: s["state"] for s in self.manager.agent_status()}
        return state.get(agent_id) == "FREE" and not self.agent_blocked(agent_id)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self.poll_once()
            except Exception:
                LOGGER.exception("native-task polling failed")

    def is_reserved(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._reserved

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": True, "ready": self._ready,
                "reserved_agents": sorted(self._reserved),
                "unacknowledged": len(self._pending_finishes),
                "offered": len(self._seen),
            }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # Called after scheduler.join(): no native side effects still in flight.
        self._finish_pending()
        try:
            self._rpc("offline")
        except TaskStoreError:
            LOGGER.warning("native-task OFFLINE failed, lease will expire")
        self._ready = False
