from __future__ import annotations

from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
import uuid

from orchestration.tasks import TaskRecord, TaskStoreError
from orchestration.event_store import EventBinding, EventStoreError
from orchestration.system_events import SystemEvent, TaskTimerSpec, TimerSpec

DEFAULT_TASK_SOCKET = '/run/cat-agent/tasks.sock'
LOGGER = logging.getLogger(__name__)


def task_record(value):
    return None if value is None else TaskRecord(**{**value, 'skills': tuple(value['skills'])})


def event_record(value):
    return None if value is None else EventBinding(**{**value, 'values': tuple(value['values'])})


class TaskRPC:
    def __init__(self, path=None):
        self.path = str(path or os.getenv('CAT_AGENT_TASK_SOCKET', DEFAULT_TASK_SOCKET))

    def call(self, target, operation, *args, **kwargs):
        request = dict(target=target, method=operation, args=args, kwargs=kwargs)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(3)
                sock.connect(self.path)
                sock.sendall((json.dumps(request, ensure_ascii=False, allow_nan=False) + '\n').encode())
                with sock.makefile('rb') as stream:
                    raw = stream.readline(1024 * 1024 + 1)
                if not raw.endswith(b'\n') or len(raw) > 1024 * 1024:
                    raise OSError('invalid Task SYSTEM response')
            response = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise TaskStoreError(f'Task SYSTEM unavailable at {self.path}: {exc}') from exc
        if not response.get('ok'):
            error = EventStoreError if response.get('type') == 'EventStoreError' else TaskStoreError
            raise error(response.get('error', 'Task SYSTEM request failed'))
        return response['result']


class RemoteTaskStore:
    def __init__(self, rpc):
        self.rpc = rpc

    def get(self, task_id):
        return task_record(self.rpc.call('tasks', 'get', task_id))

    def require(self, task_id):
        return task_record(self.rpc.call('tasks', 'require', task_id))

    def list(self):
        return tuple(task_record(t) for t in self.rpc.call('tasks', 'list'))

    def create(self, *args, **kwargs):
        return task_record(self.rpc.call('tasks', 'create', *args, **kwargs))

    def delete(self, task_id):
        return self.rpc.call('tasks', 'delete', task_id)

    def set_enabled(self, task_id, enabled):
        return task_record(self.rpc.call('tasks', 'set_enabled', task_id, enabled))

    def set_timer_period(self, task_id, period):
        return task_record(self.rpc.call('tasks', 'set_timer_period', task_id, period))

    def set_executor(self, task_id, executor):
        return task_record(self.rpc.call('tasks', 'set_executor', task_id, executor))

    def status_text(self):
        return self.rpc.call('tasks', 'status_text')


class RemoteEventStore:
    def __init__(self, rpc):
        self.rpc = rpc

    def resolve(self, *args):
        return event_record(self.rpc.call('events', 'resolve', *args))

    def register(self, *args, **kwargs):
        return event_record(self.rpc.call('events', 'register', *args, **kwargs))

    def unregister_task(self, task_id):
        return self.rpc.call('events', 'unregister_task', task_id)

    def snapshot(self):
        return tuple(event_record(e) for e in self.rpc.call('events', 'snapshot'))


class RemoteSystemRuntime:
    """A client only: no TaskStore cache, timers or local MQTT subscriptions."""
    is_remote = True

    def __init__(self, executor=None, *, rpc=None, readiness_probe=None):
        self.rpc = rpc or TaskRPC()
        self.executor = executor  # None: management access without TASK execution.
        self.task_store = RemoteTaskStore(self.rpc)
        self.event_store = RemoteEventStore(self.rpc)
        self.owner = uuid.uuid4().hex
        self._seen = set()
        self._finished = {}
        self._lock = threading.RLock()
        self._next_poll = 0.0
        self._next_warning = 0.0
        self.readiness_probe = readiness_probe
        self._reporter = None

    def set_task_handler(self, handler):
        pass  # Shared TASKs enter CORE only via a recorded execution grant.

    def arm_task_timers(self, now=None):
        pass  # Owned and armed by the service, never by a CORE restart.

    def create_task(self, *args, **kwargs):
        return task_record(self.rpc.call('system', 'create_task', *args, **kwargs))

    def create_periodic_task(self, *args, **kwargs):
        return task_record(self.rpc.call('system', 'create_periodic_task', *args, **kwargs))

    def create_external_task(self, task, binding):
        return task_record(self.rpc.call('system', 'create_external_task', task, binding))

    def start_task(self, task_id):
        return task_record(self.rpc.call('system', 'start_task', task_id))

    def stop_task(self, task_id):
        return task_record(self.rpc.call('system', 'stop_task', task_id))

    def delete_task(self, task_id):
        return self.rpc.call('system', 'delete_task', task_id)

    def set_task_period(self, task_id, period):
        return task_record(self.rpc.call('system', 'set_task_period', task_id, period))

    def task_snapshot(self):
        return self.task_store.list()

    def task_status_text(self):
        return self.task_store.status_text()

    def task_timer_snapshot(self):
        return tuple(TaskTimerSpec(**t) for t in self.rpc.call('system', 'task_timer_snapshot'))

    def timer_snapshot(self):
        return tuple(TimerSpec(**t) for t in self.rpc.call('system', 'timer_snapshot'))

    def timer_enabled(self, name):
        return self.rpc.call('system', 'timer_enabled', name)

    def timer_status_text(self):
        return self.rpc.call('system', 'timer_status_text')

    def capabilities_text(self):
        return 'TASK SYSTEM: shared; executor=auto|litert|openai (default auto)'

    def execute(self, command, body):
        return 'SYSTEM_ERROR\nUse saved TASK commands; local legacy timers are disabled'

    def submit_event(self, event):
        return self.rpc.call('system', 'submit_event', asdict(event))

    def heartbeat(self, ready, reason=''):
        return self.rpc.call('worker', 'heartbeat', owner=self.owner, executor=self.executor,
                             ready=ready, reason=reason)

    def start_worker(self, runtime_alive):
        if self.executor is None or self._reporter is not None:
            return
        from .readiness import HealthReporter
        from .ollama_usage import CheckResult
        probe = self.readiness_probe
        if self.executor == 'openai' and probe is None:
            # An OpenAI client without a configured probe is explicit-only.
            class UnknownProbe:
                def check(self):
                    return CheckResult(False, 'readiness_unknown')
            probe = UnknownProbe()
        self._reporter = HealthReporter(
            lambda result: self.heartbeat(result.ready, result.reason), runtime_alive, probe=probe)
        self._reporter.start()

    def stop_worker(self):
        if self._reporter is not None:
            self._reporter.close()
            self._reporter = None
        if self.executor is not None:
            try:
                self.rpc.call('worker', 'offline', owner=self.owner)
            except TaskStoreError:
                pass

    def flush_completions(self):
        try:
            with self._lock:
                for token, (event, outcome) in list(self._finished.items()):
                    self.rpc.call('worker', 'finish', owner=self.owner, run_id=event.run_id,
                                  offer_id=event.offer_id, outcome=outcome)
                    self._finished.pop(token)
                    self._seen.discard(token)
        except TaskStoreError:
            return False
        return True

    def poll_due(self, now=None, *, busy=False):
        if self.executor is None or time.monotonic() < self._next_poll:
            return ()
        self._next_poll = time.monotonic() + 0.5
        try:
            with self._lock:
                if not self.flush_completions():
                    return ()
                result = self.rpc.call('worker', 'pull', owner=self.owner,
                                       executor=self.executor, seen=list(self._seen))
            return () if result is None else (SystemEvent(**result),)
        except TaskStoreError as exc:
            if time.monotonic() >= self._next_warning:
                LOGGER.warning('%s', exc)
                self._next_warning = time.monotonic() + 30
            return ()

    def accepted(self, event):
        with self._lock:
            self._seen.add(event.offer_id or event.run_id)

    def begin_run(self, event):
        return self.rpc.call('worker', 'begin', owner=self.owner, run_id=event.run_id, offer_id=event.offer_id)

    def valid_run(self, event):
        return self.rpc.call('worker', 'valid', owner=self.owner, run_id=event.run_id, offer_id=event.offer_id)

    def finish_run(self, event, outcome):
        with self._lock:
            self._finished[event.offer_id or event.run_id] = (event, outcome)
        # Kept until acknowledged; losing a completion reply cannot replay a run.
        self._next_poll = 0
