"""Exclusive Unix socket ownership. No dependency on the legacy CORE."""
from __future__ import annotations

import errno
import fcntl
from pathlib import Path
import socket
import stat


class SocketOwner:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock_file = None
        self._identity: tuple[int, int] | None = None

    def _identity_at_path(self) -> tuple[int, int] | None:
        try:
            st = self.path.lstat()
        except FileNotFoundError:
            return None
        return st.st_dev, st.st_ino

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._remove_stale_socket()
        except BaseException:
            lock.close()
            raise
        self._lock_file = lock

    def _remove_stale_socket(self) -> None:
        try:
            st = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(st.st_mode):
            raise FileExistsError(f"CORE path is not a socket: {self.path}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.25)
            try:
                probe.connect(str(self.path))
            except OSError as exc:
                if exc.errno != errno.ECONNREFUSED:
                    raise
            else:
                raise OSError(errno.EADDRINUSE, f"CORE socket already active: {self.path}")
        if self._identity_at_path() != (st.st_dev, st.st_ino):
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
