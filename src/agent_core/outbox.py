"""Persistent at-least-once notification delivery, independent of inference."""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import logging
import sqlite3
import threading
import time
import uuid

LOGGER = logging.getLogger(__name__)


class NotificationOutbox:
    def __init__(self, path, sender):
        self.path = Path(path)
        self.sender = sender
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.RLock()
        self._ready = False

    @contextmanager
    def _connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous=FULL")
        if not self._ready:
            db.execute("CREATE TABLE IF NOT EXISTS notifications (id TEXT PRIMARY KEY, text TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, due REAL NOT NULL DEFAULT 0, delivered REAL, error TEXT NOT NULL DEFAULT '')")
            db.commit()
            self._ready = True
        try:
            with db:
                yield db
        finally:
            db.close()

    def enqueue(self, text):
        identifier = uuid.uuid4().hex
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO notifications(id,text) VALUES(?,?)", (identifier, text))
        return identifier

    def acknowledge(self, identifier):
        with self._lock, self._connect() as db:
            db.execute("UPDATE notifications SET delivered=?,error='' WHERE id=? AND delivered IS NULL", (time.time(), identifier))

    def pump(self, now=None):
        now = time.time() if now is None else now
        with self._lock, self._connect() as db:
            row = db.execute("SELECT id,text,attempts FROM notifications WHERE delivered IS NULL AND due<=? ORDER BY rowid LIMIT 1", (now,)).fetchone()
        if row is None:
            return False
        identifier, text, attempts = row
        try:
            accepted = self.sender(identifier, text)
            if accepted:
                self.acknowledge(identifier)
                return True
            error = "awaiting interface acknowledgement"
        except Exception as exc:
            error = str(exc)
            LOGGER.warning("Notification pending id=%s error=%s", identifier, error)
        delay = min(300, 10 * 2 ** min(attempts, 5))
        with self._lock, self._connect() as db:
            db.execute("UPDATE notifications SET attempts=attempts+1,due=?,error=? WHERE id=? AND delivered IS NULL", (now + delay, error, identifier))
        return True

    def start(self):
        with self._lock, self._connect():
            pass
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cat-agent-outbox", daemon=True)
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.pump()
            except Exception:
                LOGGER.exception("Outbox worker failed")
            self._stop.wait(0.5)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
