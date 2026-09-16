"""Persistent at-least-once notification delivery shared by all COREs."""
from __future__ import annotations

from pathlib import Path
from contextlib import contextmanager
import logging
import sqlite3
import threading
import time
import uuid

LOGGER = logging.getLogger(__name__)

DISPATCHER_LEASE_SECONDS = 1.0
PUMP_INTERVAL_SECONDS = 0.1


class NotificationOutbox:
    def __init__(self, path, sender, *, scope: str | None = None):
        self.path = Path(path)
        self.sender = sender
        self._sender_owner = getattr(sender, "__self__", None)
        if scope is None:
            owner_path = getattr(self._sender_owner, "path", None)
            scope = str(owner_path) if owner_path is not None else "default"
        self.scope = str(scope).strip() or "default"
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
            # COREs share one SQLite file. Serialize schema migration so two
            # processes cannot race while adding the routing metadata.
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS notifications ("
                "id TEXT PRIMARY KEY, text TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, "
                "due REAL NOT NULL DEFAULT 0, delivered REAL, error TEXT NOT NULL DEFAULT '', "
                "scope TEXT NOT NULL DEFAULT '')"
            )
            columns = {
                str(row[1]) for row in db.execute("PRAGMA table_info(notifications)")
            }
            if "scope" not in columns:
                db.execute(
                    "ALTER TABLE notifications ADD COLUMN scope TEXT NOT NULL DEFAULT ''"
                )
            db.execute(
                "CREATE TABLE IF NOT EXISTS dispatchers ("
                "scope TEXT PRIMARY KEY, human INTEGER NOT NULL DEFAULT 0, "
                "last_seen REAL NOT NULL DEFAULT 0)"
            )
            # Old undelivered rows have no producer scope. Give them a stable
            # scope for diagnostics; dispatch itself is elected globally.
            db.execute(
                "UPDATE notifications SET scope=? "
                "WHERE scope='' AND delivered IS NULL",
                (self.scope,),
            )
            db.commit()
            self._ready = True
        try:
            with db:
                yield db
        finally:
            db.close()

    def _human_active(self) -> bool:
        owner = self._sender_owner
        return owner is not None and getattr(owner, "_human", None) is not None

    def _heartbeat(self, db, now: float) -> None:
        db.execute(
            "INSERT INTO dispatchers(scope,human,last_seen) VALUES(?,?,?) "
            "ON CONFLICT(scope) DO UPDATE SET human=excluded.human,last_seen=excluded.last_seen",
            (self.scope, 1 if self._human_active() else 0, now),
        )

    def enqueue(self, text):
        identifier = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connect() as db:
            self._heartbeat(db, now)
            # Keep legacy retry semantics: a newly queued notification is due
            # immediately. Multi-CORE duplicate prevention is handled by the
            # dispatcher election, not by delaying the notification itself.
            db.execute(
                "INSERT INTO notifications(id,text,scope) VALUES(?,?,?)",
                (identifier, text, self.scope),
            )
        return identifier

    def acknowledge(self, identifier):
        # The human interface may live on a different CORE from the one that
        # produced the notification, so acknowledgement is intentionally global.
        with self._lock, self._connect() as db:
            db.execute(
                "UPDATE notifications SET delivered=?,error='' "
                "WHERE id=? AND delivered IS NULL",
                (time.time(), identifier),
            )

    def pump(self, now=None):
        now = time.time() if now is None else now
        live_after = now - DISPATCHER_LEASE_SECONDS
        with self._lock, self._connect() as db:
            self._heartbeat(db, now)

            # If any live CORE owns a human interface, one deterministic human
            # CORE dispatches all notifications regardless of producer.
            winner = db.execute(
                "SELECT scope FROM dispatchers "
                "WHERE human=1 AND last_seen>=? ORDER BY scope LIMIT 1",
                (live_after,),
            ).fetchone()

            # Otherwise elect one live CORE to perform Firebase fallback. This
            # prevents two COREs from pushing the same shared notification.
            if winner is None:
                winner = db.execute(
                    "SELECT scope FROM dispatchers "
                    "WHERE last_seen>=? ORDER BY scope LIMIT 1",
                    (live_after,),
                ).fetchone()

            if winner is None or str(winner[0]) != self.scope:
                return False

            row = db.execute(
                "SELECT id,text,attempts FROM notifications "
                "WHERE delivered IS NULL AND due<=? ORDER BY rowid LIMIT 1",
                (now,),
            ).fetchone()

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
            db.execute(
                "UPDATE notifications SET attempts=attempts+1,due=?,error=? "
                "WHERE id=? AND delivered IS NULL",
                (now + delay, error, identifier),
            )
        return True

    def start(self):
        now = time.time()
        with self._lock, self._connect() as db:
            self._heartbeat(db, now)
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="cat-agent-outbox", daemon=True
            )
            self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.pump()
            except Exception:
                LOGGER.exception("Outbox worker failed")
            self._stop.wait(PUMP_INTERVAL_SECONDS)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._ready:
            try:
                with self._lock, self._connect() as db:
                    db.execute("DELETE FROM dispatchers WHERE scope=?", (self.scope,))
            except Exception:
                LOGGER.exception("Failed to remove outbox dispatcher scope=%s", self.scope)
