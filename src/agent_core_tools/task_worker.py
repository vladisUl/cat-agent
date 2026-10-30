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
        self._seen: set[str] = set()
        self._reserved: set[str] = set()
        self._pending_finishes: OrderedDict[str, tuple[dict, str]] = OrderedDict()
        # A QUERY is never finished before its result is durably acknowledged.
        # During transient outages, retry publishing this immutable payload;
        # never replay either model invocation or a native tool call.
        self._pending_results: dict[str, dict] = {}
        # One immutable AGENT result per started QUERY. While the second
        # model boundary waits for Task SYSTEM, we retain only this result
        # in RAM; neither AGENT nor its tool calls are ever repeated.
        self._query_inflight: dict[str, tuple[dict, str, str, bool]] = {}
        self._query_waiting: set[str] = set()
        self._query_scheduled: set[str] = set()
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
                else:
                    LOGGER.info("native-task QUERY STORED run=%s", event["run_id"])
                    if self.on_query_result is not None:
                        try:
                            self.on_query_result(dict(result))
                        except Exception:
                            LOGGER.exception("native-task QUERY live notification failed")
                with self._lock:
                    self._pending_results.pop(token, None)
                    self._pending_finishes[token] = (event, outcome)
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

    def _finish_started(self, token: str, event: dict, agent_id: str,
                        outcome: str) -> None:
        with self._lock:
            self._reserved.discard(agent_id)
            # Never discard a started RUN before its finish is acknowledged.
            self._pending_finishes[token] = (event, outcome)

    def _complete_query(self, event: dict, agent_id: str,
                        result_text: str, agent_failed: bool) -> None:
        """Second model boundary: QUERY manager is distinct from human chat."""
        token = event["offer_id"]
        outcome = "error"
        wait_for_rpc = False
        try:
            if not self._can_run():
                outcome = "interrupted"
                return
            # TASK SYSTEM is authoritative on every cooperative model boundary.
            # An RPC outage is UNKNOWN, not revocation. Defer only the
            # validation + manager step, never replay AGENT or native tools.
            try:
                still_valid = self._rpc(
                    "valid", run_id=event["run_id"], offer_id=token,
                )
            except TaskStoreError as exc:
                LOGGER.warning("native-task QUERY VALID unavailable run=%s: %s",
                               event["run_id"], exc)
                wait_for_rpc = True
                return
            if not still_valid:
                outcome = "cancelled"
                LOGGER.info("native-task QUERY revoked before manager run=%s",
                            event["run_id"])
                return
            turn = self.manager.process_task_query(
                event["task_id"], result_text, failed=agent_failed,
            )
            if turn.kind not in ("reply", "silent"):
                raise RuntimeError("QUERY manager returned invalid kind")
            outcome = "error" if agent_failed else "done"
            text = (turn.text[:4096] if turn.kind == "reply" and turn.text.strip()
                    else f"TASK {event['task_id']}: менеджер не сформировал отчёт")
            notice = {
                "type": "task_query_result", "task_id": event["task_id"],
                "run_id": event["run_id"], "source": event.get("source", ""),
                "text": text, "agent_id": agent_id,
                "agent_ok": not agent_failed,
            }
            with self._lock:
                self._pending_results[token] = notice
            LOGGER.info("native-task QUERY MANAGER run=%s kind=%s outcome=%s",
                        event["run_id"], turn.kind, outcome)
        except Exception:
            LOGGER.exception("native-task QUERY manager failed run=%s",
                             event["run_id"])
            with self._lock:
                self._pending_results[token] = {
                    "type": "task_query_result", "task_id": event["task_id"],
                    "run_id": event["run_id"], "source": event.get("source", ""),
                    "text": f"TASK {event['task_id']}: ошибка формирования QUERY-отчёта",
                    "agent_id": agent_id, "agent_ok": False,
                }
        finally:
            with self._lock:
                self._query_scheduled.discard(token)
                if wait_for_rpc:
                    self._query_waiting.add(token)
                else:
                    self._query_inflight.pop(token, None)
                    self._query_waiting.discard(token)
            if not wait_for_rpc:
                self._finish_started(token, event, agent_id, outcome)

    def _retry_query_boundaries(self) -> None:
        """Schedule retry on the model worker; never infer on the RPC thread."""
        with self._lock:
            pending = [
                (token, self._query_inflight[token])
                for token in self._query_waiting
                if token not in self._query_scheduled and token in self._query_inflight
            ]
            for token, _ in pending:
                self._query_scheduled.add(token)
        for token, (event, agent_id, result_text, agent_failed) in pending:
            try:
                self.scheduler.submit(
                    "hardware" if event.get("source") == "mqtt" else "background",
                    lambda event=event, agent_id=agent_id, result_text=result_text,
                           agent_failed=agent_failed: self._complete_query(
                               event, agent_id, result_text, agent_failed,
                           ),
                    continuation=True,
                )
            except RuntimeError:
                with self._lock:
                    self._query_scheduled.discard(token)
                LOGGER.info("native-task QUERY retry not admitted during CORE shutdown")
                return

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
        self._retry_query_boundaries()
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
        started = False
        handed_off = False
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
            LOGGER.info("%s -> TASK %s state=%s text=%r",
                        agent_id.upper(), event["task_id"], result.state.value, result.text)
            from orchestration_tools.agents import AgentState
            agent_failed = result.state is not AgentState.FREE
            outcome = "error" if agent_failed else "done"
            LOGGER.info("native-task EXECUTED run=%s agent=%s outcome=%s",
                        event["run_id"], agent_id, outcome)
            if task["method"] == "query":
                # Yield to interactive requests BEFORE the second model call.
                # Track queued continuations: shutdown may discard the queue
                # before the manager callback starts.
                with self._lock:
                    self._query_inflight[token] = (
                        event, agent_id, result.text[:4096], agent_failed,
                    )
                try:
                    self.scheduler.submit(
                        "hardware" if event.get("source") == "mqtt" else "background",
                        lambda: self._complete_query(
                            event, agent_id, result.text[:4096], agent_failed,
                        ),
                        continuation=True,
                    )
                except Exception:
                    with self._lock:
                        self._query_inflight.pop(token, None)
                    raise
                handed_off = True
        except Exception:
            LOGGER.exception("native-task FAILED run=%s (never automatically replay)",
                             event["run_id"])
            outcome = "error"
        finally:
            with self._lock:
                if not handed_off:
                    self._reserved.discard(agent_id)
                if started and not handed_off:
                    # Finish is retried by the poll thread, NEVER the model call.
                    self._pending_finishes[token] = (event, outcome)
                elif not started:
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
                "pending_query_validations": len(self._query_waiting),
                "query_continuations": len(self._query_inflight),
                "offered": len(self._seen),
                "deferred": len(self._deferred),
                "pending_defers": len(self._pending_defers),
            }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        # Called after scheduler.join(): no native side effects still in
        # flight. Any discarded QUERY continuation is terminal: NEVER replay
        # the AGENT, even across graceful shutdown or process restart.
        with self._lock:
            orphaned = tuple(self._query_inflight.items())
            self._query_inflight.clear()
            self._query_waiting.clear()
            self._query_scheduled.clear()
        for token, (event, agent_id, _text, _failed) in orphaned:
            self._finish_started(token, event, agent_id, "interrupted")
        self._finish_pending()
        self._flush_defers()
        try:
            self._rpc("offline")
        except TaskStoreError:
            LOGGER.warning("native-task OFFLINE failed, lease will expire")
        self._ready = False
