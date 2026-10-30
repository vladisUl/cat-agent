"""Native CORE -> authoritative Task SYSTEM RPC worker (stages 5A–5D).

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
TASK_DISPLAY_REFRESH_SECONDS = 2.0
MAX_UNACKNOWLEDGED = 32


class NativeTaskWorker:
    """No local task timers, database, event hooks or model-owned RPC callbacks."""

    def __init__(
        self, scheduler, manager, *, rpc=None, agent_permissions=None,
        agent_blocked=None, on_query_result=None,
        stop_event=None, poll_seconds=POLL_SECONDS,
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
        self.on_query_result = on_query_result
        self.external_stop = stop_event
        self.clock = clock
        self.poll_seconds = poll_seconds
        self._stop = threading.Event()
        self._thread = None
        self._next_heartbeat = 0.0
        self._next_display_refresh = 0.0
        self._display_tasks: list[dict] = []
        self._display_error: str | None = None
        self._last_agent_runs: dict[str, dict] = {}
        self._seen: set[str] = set()
        self._reserved: set[str] = set()
        self._pending_finishes: OrderedDict[str, tuple[dict, str, int]] = OrderedDict()
        # A QUERY is never finished before its result is durably acknowledged.
        # During transient outages, retry publishing this immutable payload;
        # never replay either model invocation or a native tool call.
        self._pending_results: dict[str, dict] = {}
        self._deferred: dict[str, float] = {}
        self._pending_defers: OrderedDict[str, dict] = OrderedDict()
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

    def _refresh_task_display(self) -> None:
        """Read-only TASK SYSTEM telemetry, without a second timer.

        The service supplies absolute next_fire_monotonic timestamps.
        TUI interpolates the countdown using the same host monotonic clock.
        """
        now = self.clock()
        if now < self._next_display_refresh:
            return
        self._next_display_refresh = now + TASK_DISPLAY_REFRESH_SECONDS
        try:
            records = self.rpc.call("system", "task_snapshot")
            timers = self.rpc.call("system", "task_timer_snapshot")
            if not isinstance(records, list) or not isinstance(timers, list):
                raise ValueError("invalid TASK SYSTEM display snapshot")
            by_task = {
                timer["task_id"]: timer for timer in timers
                if isinstance(timer, dict) and isinstance(timer.get("task_id"), int)
            }
            display: list[dict] = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                if record.get("executor") not in ("litert", "auto"):
                    continue
                display.append({
                    "task_id": record.get("task_id"),
                    "description": record.get("description", ""),
                    "method": record.get("method", ""),
                    "executor": record.get("executor"),
                    "enabled": bool(record.get("enabled")),
                    "timer": dict(by_task[record["task_id"]])
                             if record.get("task_id") in by_task else None,
                })
        except (TaskStoreError, OSError, ValueError, TypeError, KeyError) as exc:
            with self._lock:
                self._display_tasks = []
                self._display_error = str(exc)
            LOGGER.warning("native-task DISPLAY unavailable: %s", exc)
            return
        with self._lock:
            self._display_tasks = display
            self._display_error = None

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
        for token, (event, outcome, status_code) in entries:
            with self._lock:
                result = self._pending_results.get(token)
            if result is not None:
                try:
                    persisted = self._rpc(
                        "publish_result", run_id=event["run_id"],
                        offer_id=event["offer_id"], result=result,
                    )
                except TaskStoreError as exc:
                    LOGGER.warning("native-task QUERY STORE pending run=%s: %s",
                                   event["run_id"], exc)
                    return False
                if not persisted:
                    # The authoritative RUN may have been revoked between
                    # inference and commit. Never claim the result was saved.
                    LOGGER.warning("native-task QUERY STORE rejected run=%s",
                                   event["run_id"])
                    outcome = "cancelled"
                    status_code = 1
                else:
                    LOGGER.info("native-task QUERY STORED run=%s", event["run_id"])
                    if self.on_query_result is not None:
                        try:
                            self.on_query_result(dict(result))
                        except Exception:
                            LOGGER.exception("native-task QUERY live notification failed")
                with self._lock:
                    self._pending_results.pop(token, None)
                    self._pending_finishes[token] = (event, outcome, status_code)
            try:
                acknowledged = self._rpc(
                    "finish", run_id=event["run_id"],
                    offer_id=event["offer_id"], outcome=outcome,
                    status_code=status_code,
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

    def _defer_offer(self, event: dict, *, seconds: float, reason: str) -> bool:
        """Release an UNSTARTED offer and temporarily exclude its RUN on pull.

        Task SYSTEM's existing finish(offered) atomically returns the RUN
        to pending, without changing its execution outcome.
        """
        token = event["offer_id"]
        with self._lock:
            self._deferred[event["run_id"]] = self.clock() + seconds
            self._pending_defers[token] = event
        LOGGER.info("native-task DEFER run=%s seconds=%s reason=%s",
                    event["run_id"], seconds, reason)
        return self._flush_defers()

    def _flush_defers(self) -> bool:
        with self._lock:
            entries = tuple(self._pending_defers.items())
        for token, event in entries:
            try:
                done = self._rpc("finish", run_id=event["run_id"],
                                 offer_id=token, outcome="cancelled")
            except TaskStoreError as exc:
                LOGGER.warning("native-task DEFER RPC pending run=%s: %s",
                               event["run_id"], exc)
                return False
            if done is False:
                LOGGER.info("native-task DEFER revoked run=%s", event["run_id"])
            with self._lock:
                self._pending_defers.pop(token, None)
        return True

    def _select_agent(self, task: dict) -> str | None:
        required = task.get("skills")
        if not isinstance(required, list) or not all(isinstance(s, str) for s in required):
            return None
        agents = {
            s["agent_id"]: s["state"] for s in self.manager.agent_status()
        }
        target = task.get("agent_id")
        if target is not None:
            # Native delegated schedules bind to the selected agent.
            # AGENT retains its entire registered tool set; required skills
            # are not an instruction to call any particular tool.
            if (target not in self.permissions or agents.get(target) != "FREE"
                    or not set(required).issubset(self.permissions[target])
                    or self.agent_blocked(target)):
                return None
            with self._lock:
                return target if target not in self._reserved else None
        # Never take the server lock while holding the TaskWorker lock:
        # CoreServer admission checks do the reverse (server -> worker).
        for ident, granted in self.permissions.items():
            # A saved TASK may use only tools explicitly granted to this agent.
            # Set `>` is a strict-superset comparison and ACCEPTS disjoint
            # sets; it is not an authorization check.
            if not set(required).issubset(granted) or agents.get(ident) != "FREE":
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
        if not self._finish_pending() or not self._flush_defers():
            return False  # No new execution behind unacknowledged RPC replies.
        with self._lock:
            if len(self._seen) >= MAX_UNACKNOWLEDGED:
                return False
            now = self.clock()
            self._deferred = {
                run_id: expiry for run_id, expiry in self._deferred.items()
                if expiry > now
            }
            skip_run_ids = list(self._deferred)
            seen = tuple(self._seen)
        try:
            event = self._rpc(
                "pull", executor="litert", seen=list(seen),
                skip_run_ids=skip_run_ids,
            )
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
            self._defer_offer(event, seconds=30, reason="stale_record")
            return False
        if task.get("method") not in ("task", "query"):
            self._defer_offer(event, seconds=60, reason="unknown_method")
            return False
        agent_id = self._select_agent(task)
        if agent_id is None:
            # This grant is still OFFERED. Return it to TASK SYSTEM, or it
            # will monopolize the worker's next pull forever.
            self._defer_offer(event, seconds=2, reason="agent_busy_or_not_authorized")
            return False
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
        status_code = 1
        started = False
        try:
            if not self._can_run():
                return  # Offered but never begun: service can re-offer on offline.
            # Check agent again at the actual boundary, not just at admission.
            if self._select_agent_for_reserved(agent_id) is False:
                LOGGER.info("native-task AGENT BUSY run=%s; not beginning", event["run_id"])
                return
            # BEGIN can commit on the server even when its reply is lost.
            # Record uncertain admission BEFORE RPC. FINISH(offered) requeues
            # an unstarted offer; FINISH(running) marks it interrupted.
            started = True
            try:
                allowed = self._rpc(
                    "begin", run_id=event["run_id"], offer_id=token,
                )
            except TaskStoreError:
                outcome = "interrupted"
                # If BEGIN never committed, FINISH(offered) requeues the RUN.
                # Do not immediately reclaim that same uncertain offer in
                # the poll cycle that flushes FINISH.
                with self._lock:
                    self._deferred[event["run_id"]] = self.clock() + 2.0
                LOGGER.warning("native-task BEGIN reply unknown run=%s; no model call",
                               event["run_id"])
                return
            if not allowed:
                outcome = "cancelled"
                LOGGER.info("native-task REVOKED before start run=%s", event["run_id"])
                return
            try:
                valid = self._rpc(
                    "valid", run_id=event["run_id"], offer_id=token,
                )
            except TaskStoreError:
                outcome = "interrupted"
                LOGGER.warning("native-task VALID reply unknown run=%s; no model call",
                               event["run_id"])
                return
            if not valid:
                outcome = "cancelled"
                LOGGER.info("native-task revoked before model run=%s",
                            event["run_id"])
                return
            instruction = task["text"]
            # The prompt is taken from the saved TASK record, not the IPC offer.
            LOGGER.info("TASK %s -> %s %r", event["task_id"], agent_id.upper(), instruction)
            result = self.manager.run_agent(agent_id, instruction)
            from orchestration_tools.agents import AgentState
            status_code = (result.status_code
                           if result.state is AgentState.FREE
                           else result.status_code or 1)
            LOGGER.info("%s -> TASK %s state=%s status_code=%d text=%r",
                        agent_id.upper(), event["task_id"],
                        result.state.value, status_code, result.text)
            outcome = "error" if status_code != 0 else "done"
            LOGGER.info("native-task EXECUTED run=%s agent=%s outcome=%s",
                        event["run_id"], agent_id, outcome)
            with self._lock:
                self._last_agent_runs[agent_id] = {
                    "task_id": event["task_id"], "outcome": outcome,
                    "status_code": status_code, "at_wall": time.time(),
                }
            if task["method"] == "query" and status_code == 0 and result.text.strip():
                # Only AGENT-supplied text requests delivery to the human.
                # No separate QUERY model; RUN status is always saved.
                with self._lock:
                    self._pending_results[token] = {
                        "type": "task_query_result",
                        "task_id": event["task_id"],
                        "run_id": event["run_id"],
                        "source": event.get("source", ""),
                        "agent_id": agent_id,
                        "agent_ok": True,
                        "status_code": 0,
                        "text": result.text[:4096],
                    }
        except Exception:
            LOGGER.exception("native-task FAILED run=%s (never automatically replay)",
                             event["run_id"])
            outcome = "error"
        finally:
            with self._lock:
                self._reserved.discard(agent_id)
                if started:
                    # Retry only immutable RESULT/FINISH RPC, never AGENT tools.
                    self._pending_finishes[token] = (event, outcome, status_code)
                else:
                    self._seen.discard(token)

    def _select_agent_for_reserved(self, agent_id: str) -> bool:
        state = {s["agent_id"]: s["state"] for s in self.manager.agent_status()}
        return state.get(agent_id) == "FREE" and not self.agent_blocked(agent_id)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                self._refresh_task_display()
                self.poll_once()
            except Exception:
                LOGGER.exception("native-task polling failed")

    def list_results(self, *, limit: int = 100) -> list[dict]:
        return self.rpc.call("notifications", "list", limit=limit)

    def ack_result(self, run_id: str) -> bool:
        return self.rpc.call("notifications", "ack", run_id)

    def is_reserved(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._reserved

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": True, "ready": self._ready,
                "reserved_agents": sorted(self._reserved),
                "unacknowledged": len(self._pending_finishes),
                "pending_query_stores": len(self._pending_results),
                "offered": len(self._seen),
                "deferred": len(self._deferred),
                "pending_defers": len(self._pending_defers),
                "tasks": [dict(item) for item in self._display_tasks],
                "task_display_error": self._display_error,
                "last_agent_runs": {
                    agent_id: dict(item) for agent_id, item
                    in self._last_agent_runs.items()
                },
            }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # The scheduler has drained; no model inference is in flight.
        # Persist pending notifications and FINISH without replaying AGENT.
        self._finish_pending()
        self._flush_defers()
        try:
            self._rpc("offline")
        except TaskStoreError:
            LOGGER.warning("native-task OFFLINE failed, lease will expire")
        self._ready = False
