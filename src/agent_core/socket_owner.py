"""Exclusive ownership of a CORE Unix stream socket pathname."""
from __future__ import annotations

import errno
import fcntl
from pathlib import Path
import socket
import stat


class SocketOwner:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock_file = None
        self._identity = None

    def _identity_at_path(self):
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return None
        return info.st_dev, info.st_ino

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the lock file: unlinking it would allow two different lock inodes.
        lock_file = self.path.with_name(self.path.name + ".lock").open("a")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._remove_stale_socket()
        except Exception:
            lock_file.close()
            raise
        self._lock_file = lock_file

    def _remove_stale_socket(self) -> None:
        try:
            info = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(info.st_mode):
            raise FileExistsError(f"CORE path is not a socket: {self.path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            try:
                probe.connect(str(self.path))
            except OSError as exc:
                if exc.errno != errno.ECONNREFUSED:
                    raise
            else:
                raise OSError(errno.EADDRINUSE, f"CORE socket is active: {self.path}")
        if self._identity_at_path() != (info.st_dev, info.st_ino):
            raise OSError(errno.EADDRINUSE, f"CORE socket changed: {self.path}")
        self.path.unlink()

    def bound(self) -> None:
        self._identity = self._identity_at_path()

    def release(self) -> None:
        try:
            if self._identity is not None and self._identity_at_path() == self._identity:
                self.path.unlink(missing_ok=True)
        finally:
            self._identity = None
            if self._lock_file is not None:
                self._lock_file.close()
                self._lock_file = None
