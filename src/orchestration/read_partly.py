from __future__ import annotations

import logging
import math
import shlex

from .data_paths import resolve_data_path


LOGGER = logging.getLogger(__name__)


_PART_SYSTEM_PROMPT = """Ты читаешь один фрагмент большого текстового файла.
Выполни содержательную часть задания пользователя для текущего фрагмента.
Чтение и разбиение файла уже выполнены системой.
Не пытайся читать другие фрагменты, вызывать tools или давать финальный ответ за весь файл.
Верни только результат обработки текущего фрагмента, без служебных пояснений."""


def read_partly(
    command: str,
    runtime,
    client,
    *,
    task_text: str,
    require_assignment: bool = True,
):
    """Read a text DATA file in line-safe parts and return combined model output."""
    if command.strip().split(maxsplit=1)[:1] != ["read_partly.sh"]:
        return None

    try:
        argv = shlex.split(command, posix=True)
        if len(argv) != 4 or argv[0] != "read_partly.sh" or argv[2] != "-n":
            raise ValueError("usage: read_partly.sh FILE -n COUNT")

        logical_path = argv[1]
        try:
            count = int(argv[3], 10)
        except ValueError as exc:
            raise ValueError("COUNT must be a positive integer") from exc
        if count <= 0:
            raise ValueError("COUNT must be a positive integer")

        if require_assignment and "read_partly" not in runtime.skill_names:
            raise ValueError("read_partly is not assigned to this agent")

        task = task_text.strip()
        if not task:
            raise ValueError("current task text is unavailable")

        path = resolve_data_path(runtime, logical_path, must_exist=True)
        if not path.is_file():
            raise ValueError(f"not a regular file: {logical_path}")

        text, encoding = _read_text(path)
        if not text:
            return ""

        LOGGER.info(
            "READ_PARTLY source=%s encoding=%s",
            path.name,
            encoding,
        )
        parts = _split_lines(text, count)

        fork = getattr(client, "fork", None)
        if not callable(fork):
            raise ValueError("model client does not support isolated contexts")

        try:
            child = fork("read-partly", inherit_base=False)
        except TypeError:
            child = fork("read-partly")

        base_messages = [{"role": "system", "content": _PART_SYSTEM_PROMPT}]
        results: list[str] = []

        try:
            for index, fragment in enumerate(parts, start=1):
                prompt = (
                    f"ЗАДАНИЕ:\n{task}\n\n"
                    f"ФРАГМЕНТ {index}/{len(parts)}: {path.name}\n"
                    "----- BEGIN FRAGMENT -----\n"
                    f"{fragment}\n"
                    "----- END FRAGMENT -----"
                )
                messages = [
                    *base_messages,
                    {"role": "user", "content": prompt},
                ]

                LOGGER.info(
                    "READ_PARTLY iteration=%d/%d source=%s",
                    index,
                    len(parts),
                    path.name,
                )
                response = child.chat(messages)
                result = response.content.strip()
                if result:
                    results.append(result)

                reset = getattr(child, "reset_to_base", None)
                if callable(reset):
                    reset(base_messages)
        finally:
            close = getattr(child, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    LOGGER.exception("read_partly child close failed")

        combined = "\n".join(results)
        LOGGER.info(
            "READ_PARTLY complete parts=%d source=%s chars=%d",
            len(parts),
            path.name,
            len(combined),
        )
        return combined

    except (OSError, RuntimeError, ValueError) as exc:
        return f"SYSTEM_ERROR\nread_partly.sh: {exc}"


def _read_text(path) -> tuple[str, str]:
    data = path.read_bytes()

    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return data.decode("utf-32"), "utf-32"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16"), "utf-16"
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"

    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        try:
            return data.decode("cp1251"), "cp1251"
        except UnicodeDecodeError as exc:
            raise ValueError(
                "unsupported text encoding; expected UTF-8, UTF-16 or Windows-1251"
            ) from exc


def _split_lines(text: str, count: int) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]

    part_count = min(count, len(lines))
    total_bytes = sum(len(line.encode("utf-8")) for line in lines)
    remaining_bytes = total_bytes
    remaining_parts = part_count
    target = math.ceil(remaining_bytes / remaining_parts)

    parts: list[str] = []
    current: list[str] = []
    current_bytes = 0

    for line in lines:
        current.append(line)
        current_bytes += len(line.encode("utf-8"))

        if remaining_parts > 1 and current_bytes >= target:
            parts.append("".join(current))
            remaining_bytes -= current_bytes
            remaining_parts -= 1
            current = []
            current_bytes = 0
            target = math.ceil(remaining_bytes / remaining_parts)

    if current:
        parts.append("".join(current))

    return parts
