"""Native Bash tool for the isolated LiteRT Conversation experiment.

This is intentionally a privileged capability. --enable-shell is required
by the CLI; cwd is not a sandbox. The process runs with the operator's OS
permissions. No commands are accepted from untrusted remote clients here.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Callable


LOGGER = logging.getLogger(__name__)
MAX_COMMAND_CHARS = 4096
DEFAULT_SHELL_CWD = Path("/opt/model")


@dataclass(frozen=True)
class ShellConfig:
    workdir: Path
    timeout_seconds: float = 15.0
    max_output_bytes: int = 16384

    def __post_init__(self) -> None:
        path = Path(self.workdir)
        if not path.is_dir():
            raise ValueError(f"Shell working directory does not exist: {path}")
        if not 0 < self.timeout_seconds <= 120:
            raise ValueError("Shell timeout_seconds must be in (0, 120]")
        if not 1 <= self.max_output_bytes <= 65536:
            raise ValueError("Shell max_output_bytes must be between 1 and 65536")
        object.__setattr__(self, "workdir", path.resolve())


@dataclass
class ShellGate:
    """Prevent model tools from running shell commands during BASE warm-up."""

    enabled: bool = False


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _execute(config: ShellConfig, command: str) -> dict:
    """Execute bounded Bash command, including its children, in a new group."""
    if not isinstance(command, str) or not command.strip():
        return {"ok": False, "error": "command must be a nonempty string"}
    if len(command) > MAX_COMMAND_CHARS or "\x00" in command:
        return {"ok": False, "error": "command is too long or contains NUL"}

    LOGGER.info("native-tool CALL name=shell cwd=%s command=%r", config.workdir, command)
    started = time.monotonic()
    output = bytearray()
    process = None
    error = None
    exit_code = None
    try:
        process = subprocess.Popen(
            ["/bin/bash", "-lc", command],
            cwd=config.workdir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
        assert process.stdout is not None
        deadline = started + config.timeout_seconds
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = "timeout"
                    break
                ready = selector.select(remaining)
                if not ready:
                    continue
                chunk = os.read(
                    process.stdout.fileno(),
                    min(16384, config.max_output_bytes - len(output) + 1),
                )
                if not chunk:
                    break
                space = config.max_output_bytes - len(output)
                output.extend(chunk[:space])
                if len(chunk) > space:
                    error = "output_limit_exceeded"
                    break
        if error is None:
            remaining = max(0, deadline - time.monotonic())
            try:
                exit_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                error = "timeout"
    except OSError as exc:
        error = f"process_error: {exc}"
    finally:
        if process is not None:
            if error is not None:
                _kill_process_group(process)
            elif process.poll() is None:
                _kill_process_group(process)
            if process.stdout is not None:
                process.stdout.close()

    result = {
        "ok": error is None and exit_code == 0,
        "exit_code": exit_code,
        "output": output.decode("utf-8", errors="replace"),
    }
    if error is not None:
        result["error"] = error
    elif exit_code != 0:
        result["error"] = f"exit_code_{exit_code}"
    LOGGER.info(
        "native-tool RESULT name=shell ok=%s exit_code=%s bytes=%d error=%s elapsed=%.3fs",
        result["ok"], exit_code, len(output), result.get("error"), time.monotonic() - started,
    )
    return result


def create_shell_tool(
    config: ShellConfig, *, gate: ShellGate | None = None,
) -> Callable[[str], dict]:
    """Return one function for LiteRT's automatic native tool registration."""

    def shell(command: str) -> dict:
        """Выполнить Bash-команду в рабочем каталоге и получить её вывод.

        Args:
            command (str): Команда Bash, включая допустимые аргументы и конвейеры.
        """
        LOGGER.info("native-tool DISPATCH name=shell arg_type=%s value=%r",
                    type(command).__name__, command)
        if gate is not None and not gate.enabled:
            return {"ok": False, "error": "shell disabled during BASE warm-up"}
        return _execute(config, command)

    return shell
