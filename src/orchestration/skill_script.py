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


SKILL_SILENT = object()


def run_skill_script(command: str, runtime, client):
    """Execute an assigned <skill>.sh scenario from the workspace root.

    Returns None when command isn't an assigned skill script. SKILL_SILENT
    means the assigned scenario completed successfully without a result.

    Text after the scenario name is opaque pipeline input. The first step gets
    it on stdin; every following text-producing step gets the previous step's
    output. A final read_pic.sh returns its normal multimodal payload.
    """
    stripped_command = command.strip()
    if not stripped_command:
        return None

    parts = stripped_command.split(maxsplit=1)
    script_name = parts[0]
    input_text = parts[1] if len(parts) == 2 else ""

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
        return (
            f"SYSTEM_ERROR\n{script_name}: skill script exceeds "
            f"{runtime.max_file_bytes} bytes"
        )

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"SYSTEM_ERROR\n{script_name}: cannot read skill script: {exc}"

    steps = [
        (line_number, raw.strip())
        for line_number, raw in enumerate(text.splitlines(), 1)
        if raw.strip() and not raw.strip().startswith("#")
    ]
    if not steps:
        return f"SYSTEM_ERROR\n{script_name}: skill script is empty"

    stream = input_text

    for step_index, (line_number, stripped) in enumerate(steps):
        try:
            tokens = shlex.split(stripped, posix=True)
        except ValueError as exc:
            return (
                f"SYSTEM_ERROR\n{script_name}:{line_number}: parse error: {exc}"
            )
        if not tokens:
            continue

        if tokens[0] == "read_pic.sh":
            picture = read_picture(
                stripped,
                runtime,
                client,
                require_assignment=False,
                input_text=stream,
            )
            if picture is None:
                return (
                    f"SYSTEM_ERROR\n{script_name}:{line_number}: "
                    "invalid read_pic.sh command"
                )
            if isinstance(picture, str) and picture.startswith("SYSTEM_ERROR\n"):
                return (
                    f"SYSTEM_ERROR\n{script_name}:{line_number}: "
                    f"{picture.splitlines()[-1]}"
                )
            if step_index != len(steps) - 1:
                return (
                    f"SYSTEM_ERROR\n{script_name}:{line_number}: "
                    "read_pic.sh produces a non-text result and must be the final step"
                )
            LOGGER.info(
                "skill-script %s line=%d internal=read_pic.sh",
                script_name,
                line_number,
            )
            return picture

        internal = runtime.execute_internal_command(
            stripped,
            require_assignment=False,
            input_text=stream,
        )
        if internal is not None:
            LOGGER.info(
                "skill-script %s line=%d internal=%s exit=%d",
                script_name,
                line_number,
                tokens[0],
                internal.exit_code,
            )
            if not internal.ok:
                if internal.exit_code == 124:
                    uncertain.add(command)
                    runtime._uncertain_commands = uncertain
                rendered = runtime.format_result(internal)
                return (
                    f"SYSTEM_ERROR\n{script_name}:{line_number} failed\n"
                    f"{rendered}"
                )
            stream = runtime.format_result(internal).strip()
            continue

        result = _execute_external(
            stripped,
            tokens,
            runtime,
            input_text=stream,
        )
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
            return (
                f"SYSTEM_ERROR\n{script_name}:{line_number} failed\n"
                f"{rendered}"
            )
        stream = result.stdout.strip()

    if stream:
        return stream
    return SKILL_SILENT


def _execute_external(
    command: str,
    tokens: list[str],
    runtime,
    *,
    input_text: str,
) -> CommandResult:
    name = tokens[0]
    if Path(name).name != name or "/" in name:
        return _error(
            runtime,
            command,
            name,
            126,
            "command name must be a file in workspace root",
            "invalid_command",
        )

    executable = runtime.root / name
    if executable.is_symlink():
        return _error(
            runtime,
            command,
            name,
            126,
            f"{name}: symlink is not permitted",
            "symlink",
        )
    if not executable.exists():
        return _error(
            runtime,
            command,
            name,
            127,
            f"{name}: command not found",
            "command_not_found",
        )
    if not executable.is_file():
        return _error(
            runtime,
            command,
            name,
            126,
            f"{name}: not a regular file",
            "not_file",
        )
    if not os.access(executable, os.X_OK):
        return _error(
            runtime,
            command,
            name,
            126,
            f"{name}: file is not executable",
            "not_executable",
        )

    try:
        resolved_args = [
            token
            if token.startswith("-")
            else str(resolve_data_path(runtime, token))
            for token in tokens[1:]
        ]
    except ValueError as exc:
        return _error(
            runtime,
            command,
            name,
            126,
            str(exc),
            "invalid_data_path",
        )

    argv = [str(executable), *resolved_args]
    try:
        completed = run_process(
            argv,
            cwd=runtime.root,
            timeout=runtime.timeout_seconds,
            output_limit=runtime.max_file_bytes,
            input_text=input_text,
        )
    except FileNotFoundError:
        return _error(
            runtime,
            command,
            name,
            127,
            f"{name}: command not found",
            "command_not_found",
        )
    except OSError as exc:
        return _error(
            runtime,
            command,
            name,
            126,
            f"{name}: {exc}",
            "exec_error",
        )

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
