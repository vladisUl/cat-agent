from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex

from .command_runtime import CommandResult
from .data_paths import resolve_data_path
from .image_tool import read_picture
from .process_runner import run_process


LOGGER = logging.getLogger(__name__)


def run_skill_script(command: str, runtime, client):
    """Execute an assigned <skill>.sh scenario from the workspace root.

    Returns None when command isn't an assigned skill script. The scenario is
    our own line-oriented format, not bash. Ordinary lines invoke executable
    files from the workspace root; read_pic.sh is an internal opcode.
    """
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return None
    if len(argv) != 1:
        return None

    script_name = argv[0]
    script_path = Path(script_name)
    if (
        script_path.name != script_name
        or script_path.suffix != ".sh"
        or script_path.stem not in runtime.skill_names
    ):
        return None

    uncertain = getattr(runtime, "_uncertain_commands", set())
    if command in uncertain:
        return (
            "SYSTEM_ERROR\n"
            f"{script_name}: previous attempt timed out with uncertain side effects; "
            "automatic retry refused"
        )

    path = runtime.root / script_name
    if path.is_symlink():
        return f"SYSTEM_ERROR\n{script_name}: symlink is not permitted"
    if not path.exists():
        return f"SYSTEM_ERROR\n{script_name}: skill script not found"
    if not path.is_file():
        return f"SYSTEM_ERROR\n{script_name}: skill script is not a regular file"
    if path.stat().st_size > runtime.max_file_bytes:
        return f"SYSTEM_ERROR\n{script_name}: skill script exceeds {runtime.max_file_bytes} bytes"

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"SYSTEM_ERROR\n{script_name}: cannot read skill script: {exc}"

    image_result = None
    executed = 0
    for line_number, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError as exc:
            return f"SYSTEM_ERROR\n{script_name}:{line_number}: parse error: {exc}"
        if not tokens:
            continue

        if tokens[0] == "read_pic.sh":
            picture = read_picture(
                stripped,
                runtime,
                client,
                require_assignment=False,
            )
            if picture is None:
                return f"SYSTEM_ERROR\n{script_name}:{line_number}: invalid read_pic.sh command"
            if isinstance(picture, str) and picture.startswith("SYSTEM_ERROR\n"):
                return f"SYSTEM_ERROR\n{script_name}:{line_number}: {picture.splitlines()[-1]}"
            if image_result is not None:
                return f"SYSTEM_ERROR\n{script_name}:{line_number}: only one read_pic.sh is supported"
            image_result = picture
            executed += 1
            LOGGER.info("skill-script %s line=%d internal=read_pic.sh", script_name, line_number)
            continue

        result = _execute_external(stripped, tokens, runtime)
        LOGGER.info(
            "skill-script %s line=%d command=%s exit=%d",
            script_name,
            line_number,
            tokens[0],
            result.exit_code,
        )
        if not result.ok:
            if result.exit_code == 124:
                uncertain.add(command)
                runtime._uncertain_commands = uncertain
            rendered = runtime.format_result(result)
            return f"SYSTEM_ERROR\n{script_name}:{line_number} failed\n{rendered}"
        executed += 1

    if executed == 0:
        return f"SYSTEM_ERROR\n{script_name}: skill script is empty"
    if image_result is not None:
        return image_result
    return f"SYSTEM_OK\n{script_name}: completed"


def _execute_external(command: str, tokens: list[str], runtime) -> CommandResult:
    name = tokens[0]
    if Path(name).name != name or "/" in name:
        return _error(runtime, command, name, 126, "command name must be a file in workspace root", "invalid_command")

    executable = runtime.root / name
    if executable.is_symlink():
        return _error(runtime, command, name, 126, f"{name}: symlink is not permitted", "symlink")
    if not executable.exists():
        return _error(runtime, command, name, 127, f"{name}: command not found", "command_not_found")
    if not executable.is_file():
        return _error(runtime, command, name, 126, f"{name}: not a regular file", "not_file")
    if not os.access(executable, os.X_OK):
        return _error(runtime, command, name, 126, f"{name}: file is not executable", "not_executable")

    try:
        resolved_args = [
            str(resolve_data_path(runtime, token)) if token.startswith("/") else token
            for token in tokens[1:]
        ]
    except ValueError as exc:
        return _error(runtime, command, name, 126, str(exc), "invalid_data_path")

    argv = [str(executable), *resolved_args]
    try:
        completed = run_process(
            argv,
            cwd=runtime.root,
            timeout=runtime.timeout_seconds,
            output_limit=runtime.max_file_bytes,
        )
    except FileNotFoundError:
        return _error(runtime, command, name, 127, f"{name}: command not found", "command_not_found")
    except OSError as exc:
        return _error(runtime, command, name, 126, f"{name}: {exc}", "exec_error")

    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        cwd=runtime.root,
        operation=name,
        metadata={
            "executable": str(executable),
            **(
                {"error_code": "timeout", "outcome": "uncertain"}
                if completed.returncode == 124
                else {}
            ),
        },
    )


def _error(runtime, command, operation, exit_code, stderr, code):
    return CommandResult(
        command=command,
        exit_code=exit_code,
        stdout="",
        stderr=stderr,
        cwd=runtime.root,
        operation=operation,
        metadata={"error_code": code},
    )
