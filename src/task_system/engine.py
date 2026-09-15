from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import fcntl
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from orchestration.tasks import TaskStore, TaskRecord, TaskStoreError, DEFAULT_TASK_FILE
from orchestration.event_store import EventStore, EventBinding, DEFAULT_EVENT_FILE
from orchestration.system_events import SystemRuntime, SystemEvent, TaskTimerSpec, TimerSpec

LOGGER = logging.getLogger(__name__)

TERMINAL = {"done", "error", "cancelled", "interrupted"}


def process_identity(pid: int) -> str:
    """PID plus boot and start time; a reused PID is not the previous worker."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return ""
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return f"{pid}:{boot}:{fields[19]}"
    except (OSError, IndexError):
        return ""


def process_alive(identity: str) -> bool:
    try:
        return bool(identity) and process_identity(int(identity.split(":", 1)[0])) == identity
    except ValueError:
        return False


class OwnedTasks(TaskStore):
    def __init__(self, path, *, import_legacy=True):
        self.import_legacy = import_legacy
        super().__init__(path)

    def reload(self):
        if self.import_legacy:
            super().reload()
        else:
            self._tasks = {}

    def _save(self):
        # Engine commits registry, timers, bindings and runs in ONE transaction.
        pass


class OwnedEvents(EventStore):
    def __init__(self, path, *, import_legacy=True):
        self.import_legacy = import_legacy
        super().__init__(path)

    def reload(self):
        if self.import_legacy:
            super().reload()
        else:
            self._bindings = {}

    def _save_locked(self):
        pass


class TaskEngine:
    def __init__(self, path: Path, *, task_file=DEFAULT_TASK_FILE,
                 event_file=DEFAULT_EVENT_FILE, alive=process_alive):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = path.with_name(path.name + ".lock").open("a")
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self._file_lock.close()
            raise
        self._lock = threading.RLock()
        self.alive = alive
        self.db = None
        try:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)")
            row = self.db.execute("SELECT body FROM state WHERE id=1").fetchone()
            # Legacy files are imported once and never written by CORE again.
            self.tasks = OwnedTasks(Path(task_file), import_legacy=row is None)
            self.events = OwnedEvents(Path(event_file), import_legacy=row is None)
            self.runtime = SystemRuntime(self.tasks)
            self.runs = {}
            if row is not None:
                self._restore(json.loads(row[0]))
            else:
                self.runtime.arm_task_timers()
                self._save()
        except Exception:
            self.close()
            raise

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
        self._file_lock.close()

    def _snapshot(self, *, wall=True):
        offset = time.time() - time.monotonic() if wall else 0.0
        def timers(items):
            result = []
            for item in items:
                data = asdict(item)
                deadline = data.pop('next_fire_monotonic')
                data['next_fire_wall'] = None if deadline is None else deadline + offset
                result.append(data)
            return result
        return dict(tasks=[asdict(t) for t in self.tasks.list()],
                    events=[asdict(e) for e in self.events.snapshot()],
                    timers=timers(self.runtime.task_timer_snapshot()),
                    legacy_timers=timers(self.runtime.timer_snapshot()), runs=self.runs)

    def _save(self):
        body = json.dumps(self._snapshot(), ensure_ascii=False, allow_nan=False)
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES(1,?)", (body,))

    def _restore(self, data):
        self.tasks._tasks = {t['task_id']: TaskRecord(**{**t, 'skills': tuple(t['skills'])}) for t in data['tasks']}
        with self.events._lock:
            self.events._bindings = {e['name']: EventBinding(**{**e, 'values': tuple(e['values'])}) for e in data['events']}
        offset = time.time() - time.monotonic()
        def timers(items, cls, key):
            result = {}
            for item in items:
                item = dict(item)
                deadline = item.pop('next_fire_wall')
                item['next_fire_monotonic'] = None if deadline is None else deadline - offset
                timer = cls(**item)
                result[getattr(timer, key)] = timer
            return result
        self.runtime._task_timers = timers(data['timers'], TaskTimerSpec, 'task_id')
        self.runtime._timers = timers(data.get('legacy_timers', []), TimerSpec, 'name')
        self.runs = data['runs']

    def _current(self, run):
        task = self.tasks.get(run['event']['task_id'])
        return task is not None and task.enabled and task.generation == run['event']['task_generation']

    def _queue(self, event):
        if event.task_id is None:
            raise TaskStoreError("shared scheduling requires a saved TASK")
        task = self.tasks.get(event.task_id)
        if task is None or not task.enabled or task.generation != event.task_generation:
            return None
        if event.source == 'timer' and any(r['event']['task_id'] == task.task_id and r['event']['task_generation'] == task.generation and r['state'] not in TERMINAL for r in self.runs.values()):
            return None  # Coalesce missed timer ticks; do not build an offline backlog.
        if sum(r['state'] not in TERMINAL for r in self.runs.values()) >= 1024:
            raise TaskStoreError("Task SYSTEM pending queue is full")
        identifier = uuid.uuid4().hex
        payload = asdict(event)
        payload['run_id'] = identifier
        self.runs[identifier] = dict(event=payload, executor=task.executor, state='pending',
                                     owner='', peer='', outcome='')
        LOGGER.info('TASK queued id=%s run=%s executor=%s source=%s', task.task_id, identifier, task.executor, event.source)
        return identifier

    def _reconcile(self):
        for run in self.runs.values():
            if run['state'] in {'pending', 'offered'} and not self._current(run):
                run['state'] = 'cancelled'
            elif run['state'] in {'offered', 'running'} and not self.alive(run['peer']):
                # Never replay potentially completed side effects after a worker crash.
                run['state'] = 'pending' if run['state'] == 'offered' else 'interrupted'
                LOGGER.warning('TASK worker disappeared run=%s state=%s', run['event']['run_id'], run['state'])
                run['owner'] = run['peer'] = ''
        completed = [key for key, run in self.runs.items() if run['state'] in TERMINAL]
        for key in completed[:-100]:
            del self.runs[key]

    def consume_mqtt(self, binding, value):
        with self._lock:
            if self.events.resolve(binding.source, binding.name) is not binding:
                return None  # A deleted/recreated binding must not receive old payloads.
            return self.call('system', 'external_event', (binding.source, binding.name, value))

    def tick(self, now=None):
        with self._lock:
            before = json.dumps(self._snapshot())
            signature = json.dumps(self._snapshot(wall=False))
            try:
                self._reconcile()
                for event in self.runtime.poll_due(now):
                    self._queue(event)
                if signature != json.dumps(self._snapshot(wall=False)):
                    self._save()
            except Exception:
                self._restore(json.loads(before))
                raise

    def call(self, target, method, args=(), kwargs=None, *, peer=''):
        with self._lock:
            before = json.dumps(self._snapshot())
            signature = json.dumps(self._snapshot(wall=False))
            try:
                result = self._dispatch(target, method, args, kwargs or {}, peer)
                if signature != json.dumps(self._snapshot(wall=False)):
                    self._save()
                return deepcopy(result)
            except Exception:
                self._restore(json.loads(before))
                raise

    def _dispatch(self, target, method, args, kwargs, peer):
        if target == 'tasks':
            if method in {'get', 'require', 'list'}:
                return getattr(self.tasks, method)(*args, **kwargs)
            if method == 'status_text':
                return self.status_text()
            if method == 'create':
                task = self.tasks.create(*args, **kwargs)
                if task.timer_period_seconds is not None:
                    self.runtime._task_timers[task.task_id] = TaskTimerSpec(
                        task.task_id, task.timer_period_seconds, task.enabled,
                        time.monotonic() + task.timer_period_seconds if task.enabled else None)
                return task
            if method == 'delete':
                return self._delete(*args, **kwargs)
            if method == 'set_enabled':
                return (self.runtime.start_task if args[1] else self.runtime.stop_task)(args[0])
            if method == 'set_timer_period':
                return self.runtime.set_task_period(*args, **kwargs)
            if method == 'set_executor':
                return self.tasks.set_executor(*args, **kwargs)
        if target == 'events':
            if method in {'resolve', 'snapshot', 'unregister_task'}:
                return getattr(self.events, method)(*args, **kwargs)
            if method == 'register':
                self.tasks.require(args[0])
                return self.events.register(*args, **kwargs)
        if target == 'system':
            if method == 'task_status_text':
                return self.status_text()
            if method == 'delete_task':
                return self._delete(*args, **kwargs)
            if method == 'create_external_task':
                task_kwargs, binding_kwargs = args
                task = self.tasks.create(**task_kwargs)
                self.events.register(task.task_id, task.description, **binding_kwargs)
                return task
            if method in {'create_task', 'create_periodic_task', 'start_task', 'stop_task',
                          'set_task_period', 'task_snapshot', 'task_timer_snapshot',
                          'timer_snapshot', 'timer_enabled', 'timer_status_text', 'capabilities_text'}:
                return getattr(self.runtime, method)(*args, **kwargs)
            if method == 'ping':
                return 'TASK_SYSTEM_READY'
            if method == 'run_task':
                task = self.tasks.require(args[0])
                return self._queue(SystemEvent('manual', f'task:{task.task_id}', '', time.monotonic(), task.task_id, task.generation))
            if method == 'runs':
                return list(self.runs.values())
            if method == 'external_event':
                source, name, value = args
                binding = self.events.resolve(source, name)
                if binding is None:
                    return None
                task = self.tasks.get(binding.task_id)
                if task is None:
                    return None
                return self._queue(SystemEvent(source, name, value or '', time.monotonic(), task.task_id, task.generation))
            if method == 'submit_event':
                return self._queue(SystemEvent(**args[0]))
        if target == 'worker':
            return self._worker(method, kwargs, peer)
        raise TaskStoreError(f"unsupported Task SYSTEM operation: {target}.{method}")

    def _delete(self, task_id):
        deleted = self.runtime.delete_task(task_id)
        self.events.unregister_task(task_id)
        return deleted

    def status_text(self):
        lines = []
        for task in self.tasks.list():
            live = [r for r in self.runs.values() if r['event']['task_id'] == task.task_id and r['state'] not in TERMINAL]
            states = ','.join(sorted({r['state'] for r in live})) or 'idle'
            lines.append(f"TASK {task.task_id} executor={task.executor} "
                         f"{'started' if task.enabled else 'stopped'} period={task.timer_period_seconds} "
                         f"runs={states} {task.description}")
        return '\n'.join(lines) or 'no tasks'

    def _worker(self, method, kw, peer):
        owner = kw['owner']
        if not peer or not owner:
            raise TaskStoreError('worker identity required')
        self._reconcile()
        if method == 'pull':
            executor = kw['executor']
            self.tasks._validate_executor(executor)
            seen = set(kw.get('seen', []))
            for run in self.runs.values():
                if run['state'] == 'offered' and run['owner'] == owner and run['peer'] == peer and run['event']['run_id'] not in seen:
                    return run['event']
            occupied = {r['event']['task_id'] for r in self.runs.values() if r['state'] in {'offered', 'running'}}
            for run in self.runs.values():
                if run['state'] == 'pending' and run['executor'] == executor and run['event']['task_id'] not in occupied:
                    run.update(state='offered', owner=owner, peer=peer)
                    LOGGER.info('TASK offered id=%s run=%s executor=%s', run['event']['task_id'], run['event']['run_id'], executor)
                    return run['event']
            return None
        run = self.runs.get(kw['run_id'])
        if run is None or run['owner'] != owner or run['peer'] != peer:
            return False
        if method in {'begin', 'valid'}:
            if run['state'] not in {'offered', 'running'} or not self._current(run):
                return False
            if method == 'begin':
                run['state'] = 'running'
                LOGGER.info('TASK running id=%s run=%s executor=%s', run['event']['task_id'], kw['run_id'], run['executor'])
            return True
        if method == 'finish':
            if run['state'] in TERMINAL:
                return True
            outcome = kw.get('outcome', 'done')
            if run['state'] == 'offered' and self._current(run):
                run.update(state='pending', owner='', peer='')
            else:
                run.update(state=outcome if outcome in TERMINAL else 'done', outcome=outcome)
            LOGGER.info('TASK finished id=%s run=%s state=%s', run['event']['task_id'], kw['run_id'], run['state'])
            return True
        raise TaskStoreError(f'unknown worker operation: {method}')
