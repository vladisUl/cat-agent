from __future__ import annotations

import logging
import os
from pathlib import Path
import shlex
import tempfile

from .command_runtime import CommandResult
from .data_paths import resolve_data_path


LOGGER = logging.getLogger(__name__)


_CYCLIC_SYSTEM_PROMPT = """Ты обработчик одного фрагмента данных.
Выполни только содержательную часть указанного задания для текущего фрагмента.
Если исходное задание говорит использовать cyclic_process, считай этот шаг уже выполненным системой.
Не пытайся читать другие фрагменты, вызывать tools или продолжать цикл самостоятельно.
Ответь только результатом обработки текущего фрагмента, без служебных пояснений."""


def execute_cyclic_process(
    command: str,
    runtime,
    client,
    *,
    task_text: str,
) -> CommandResult:
    """Process PREFIX_1..PREFIX_N in isolated model contexts.

    This is a pseudo-script primitive. It is intentionally independent of
    MANAGER/AGENT roles: the caller supplies only the current task text, runtime
    DATA root and model client.
    """

    def fail(message: str, code: str = "invalid_arguments") -> CommandResult:
        return CommandResult(
            command=command,
            exit_code=2,
            stdout="",
            stderr=message,
            cwd=runtime.root,
            operation="cyclic_process",
            metadata={"error_code": code},
        )

    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        return fail(f"cyclic_process: {exc}", "parse_error")

    if (
        len(tokens) != 4
        or tokens[0] != "cyclic_process"
        or tokens[2] != "-n"
    ):
        return fail(
            "cyclic_process: usage: cyclic_process PREFIX -n COUNT"
        )

    prefix = tokens[1]
    try:
        count = int(tokens[3], 10)
    except ValueError:
        return fail("cyclic_process: COUNT must be a positive integer")
    if count <= 0:
        return fail("cyclic_process: COUNT must be a positive integer")

    task = task_text.strip()
    if not task:
        return fail(
            "cyclic_process: current task text is unavailable",
            "missing_task",
        )

    logical_prefix = prefix[1:] if prefix.startswith("/") else prefix
    if not logical_prefix:
        return fail("cyclic_process: PREFIX must not be empty")

    prefix_path = Path(logical_prefix)
    suffix = prefix_path.suffix
    stem = prefix_path.stem
    parent = prefix_path.parent

    part_names = [
        (parent / f"{stem}_{index}{suffix}").as_posix()
        for index in range(1, count + 1)
    ]
    output_name = (parent / f"{stem}_out.txt").as_posix()

    try:
        part_paths = [
            resolve_data_path(runtime, name, must_exist=True)
            for name in part_names
        ]
        output_path = resolve_data_path(runtime, output_name)
    except (OSError, ValueError) as exc:
        return fail(f"cyclic_process: {exc}", "invalid_data_path")

    if output_path.exists() and not output_path.is_file():
        return fail(
            f"cyclic_process: output is not a regular file: {output_name}",
            "invalid_output",
        )

    fork = getattr(client, "fork", None)
    if not callable(fork):
        return fail(
            "cyclic_process: model client does not support isolated contexts",
            "unsupported_client",
        )

    try:
        try:
            child = fork("cyclic-process", inherit_base=False)
        except TypeError:
            child = fork("cyclic-process")
    except Exception as exc:
        LOGGER.exception("cyclic_process client fork failed")
        return fail(
            f"cyclic_process: cannot create isolated model context: {exc}",
            "fork_failed",
        )

    base_messages = [
        {"role": "system", "content": _CYCLIC_SYSTEM_PROMPT},
    ]

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temp_path = Path(temp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            for index, (name, path) in enumerate(
                zip(part_names, part_paths),
                start=1,
            ):
                fragment = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                prompt = (
                    f"ЗАДАНИЕ:\n{task}\n\n"
                    f"ФРАГМЕНТ {index}/{count}: {name}\n"
                    "----- BEGIN FRAGMENT -----\n"
                    f"{fragment}\n"
                    "----- END FRAGMENT -----"
                )
                messages = [
                    *base_messages,
                    {"role": "user", "content": prompt},
                ]

                LOGGER.info(
                    "CYCLIC_PROCESS iteration=%d/%d source=%s",
                    index,
                    count,
                    name,
                )
                response = child.chat(messages)
                result = response.content.strip()
                if result:
                    output.write(result)
                    output.write("\n")

                reset = getattr(child, "reset_to_base", None)
                if callable(reset):
                    reset(base_messages)

        os.replace(temp_path, output_path)
    except Exception as exc:
        LOGGER.exception("cyclic_process failed")
        return fail(
            f"cyclic_process: {exc}",
            "iteration_failed",
        )
    finally:
        temp_path.unlink(missing_ok=True)
        close = getattr(child, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.exception("cyclic_process child close failed")

    LOGGER.info(
        "CYCLIC_PROCESS complete parts=%d output=%s",
        count,
        output_name,
    )
    return CommandResult(
        command=command,
        exit_code=0,
        stdout=output_name + "\n",
        stderr="",
        cwd=runtime.root,
        operation="cyclic_process",
        metadata={
            "prefix": prefix,
            "count": count,
            "output": output_name,
        },
    )
