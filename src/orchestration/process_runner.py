"""Bounded process output and process-group timeouts; no command retries."""
from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time


def run_process(argv, *, cwd, timeout, output_limit):
    process = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = set()
    deadline = time.monotonic() + timeout
    expired = False
    with selectors.DefaultSelector() as selector:
        for name in chunks:
            pipe = getattr(process, name)
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        try:
            while selector.get_map() or process.poll() is None:
                if time.monotonic() >= deadline:
                    expired = True
                    break
                for key, _ in selector.select(min(0.1, max(0, deadline - time.monotonic()))):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    target = chunks[key.data]
                    room = max(0, output_limit - len(target))
                    target.extend(data[:room])
                    if len(data) > room:
                        truncated.add(key.data)
            if expired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
                # Kill remaining descendants even if the bash leader already exited.
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
            for pipe in (process.stdout, process.stderr):
                pipe.close()
    result = {name: bytes(data).decode("utf-8", errors="replace") for name, data in chunks.items()}
    for name in truncated:
        result[name] += "\n[output truncated]"
    if expired:
        result["stderr"] += "\ncommand timed out; process group terminated; side effects may be partial, do not retry automatically"
    return subprocess.CompletedProcess(argv, 124 if expired else code, result["stdout"], result["stderr"])
