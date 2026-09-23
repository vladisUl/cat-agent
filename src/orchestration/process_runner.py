"""Bounded process I/O and process-group timeouts; no command retries."""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time


def run_process(argv, *, cwd, timeout, output_limit, input_text=None):
    """Run one process with bounded stdout/stderr and optional UTF-8 stdin.

    When input_text is not None, stdin is a pipe owned by this function and is
    closed after the complete input has been written.  Stdin is handled by the
    same non-blocking selector loop as stdout/stderr so a child that writes
    before consuming all input cannot deadlock the runner.
    """
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = set()
    deadline = time.monotonic() + timeout
    expired = False
    stdin_data = b"" if input_text is None else input_text.encode("utf-8")
    stdin_offset = 0

    with selectors.DefaultSelector() as selector:
        for name in chunks:
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, ("read", name))

        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            if stdin_data:
                selector.register(process.stdin, selectors.EVENT_WRITE, ("write", "stdin"))
            else:
                process.stdin.close()

        try:
            while selector.get_map() or process.poll() is None:
                if time.monotonic() >= deadline:
                    expired = True
                    break

                wait = min(0.1, max(0, deadline - time.monotonic()))
                for key, _ in selector.select(wait):
                    direction, name = key.data

                    if direction == "write":
                        try:
                            written = os.write(
                                key.fileobj.fileno(),
                                stdin_data[stdin_offset:],
                            )
                        except (BrokenPipeError, OSError):
                            written = 0
                            try:
                                selector.unregister(key.fileobj)
                            except KeyError:
                                pass
                            key.fileobj.close()
                            continue

                        stdin_offset += written
                        if stdin_offset >= len(stdin_data):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                        continue

                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    target = chunks[name]
                    room = max(0, output_limit - len(target))
                    target.extend(data[:room])
                    if len(data) > room:
                        truncated.add(name)

            if expired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
                # Kill remaining descendants even if the process leader exited.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            code = process.wait()
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None and not pipe.closed:
                    pipe.close()

    result = {
        name: bytes(data).decode("utf-8", errors="replace")
        for name, data in chunks.items()
    }
    for name in truncated:
        result[name] += "\n[output truncated]"
    if expired:
        result["stderr"] += (
            "\ncommand timed out; process group terminated; side effects may be "
            "partial, do not retry automatically"
        )
    return subprocess.CompletedProcess(
        argv,
        124 if expired else code,
        result["stdout"],
        result["stderr"],
    )
