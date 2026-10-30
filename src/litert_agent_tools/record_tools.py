"""Audited, narrowly-scoped native tool for outdoor MQTT temperature logging.

Enabled only for AGENTs explicitly granted privileged shell capability,
but never runs a shell. Filenames are single safe .txt basenames under
the configured native working directory.
"""
from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import re
import stat

from .config import MqttConfig
from .mqtt_tools import _read_measurement, TOPICS
from .shell_tools import ShellGate
from .tool_audit import record_task_tool_result


LOGGER = logging.getLogger(__name__)
RECORD_TOOL_NAME = "record_outdoor_weather"
FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.txt\Z", re.ASCII)
INSTRUCTION_FILENAME = re.compile(
    r"(?<![\w./\\-])([A-Za-z0-9][A-Za-z0-9_.-]*\.txt)(?![\w./\\-])",
    re.ASCII | re.IGNORECASE,
)


def valid_record_filename(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) <= 80
        and ".." not in value and bool(FILENAME.fullmatch(value))
    )


def extract_record_filename(instruction: str) -> str | None:
    matches = set(INSTRUCTION_FILENAME.findall(instruction))
    if len(matches) != 1:
        return None
    candidate = next(iter(matches))
    return candidate if valid_record_filename(candidate) else None


def create_record_outdoor_weather_tool(
    config: MqttConfig, workdir: Path, *, gate: ShellGate,
):
    """Register an atomic single-tool MQTT-read + append operation."""
    root = Path(workdir).resolve()
    if not root.is_dir():
        raise ValueError(f"Record directory does not exist: {root}")

    def record_outdoor_weather(file_name: str) -> dict:
        """Измерить температуру на улице и ДОПИСАТЬ строку в текстовый файл.

        Файл создаётся автоматически при первой записи; при последующих
        вызовах новая строка дописывается. Разрешено только имя .txt без пути
        в настроенном рабочем каталоге.

        Args:
            file_name (str): Имя файла, например temp.txt.
        """
        LOGGER.info("native-tool CALL name=%s file_name=%r", RECORD_TOOL_NAME, file_name)

        def finish(result: dict) -> dict:
            record_task_tool_result(RECORD_TOOL_NAME, result)
            LOGGER.info("native-tool RESULT name=%s result=%r", RECORD_TOOL_NAME, result)
            return result

        if not gate.enabled:
            return finish({"ok": False, "error": "file logging disabled during BASE warm-up"})
        if not valid_record_filename(file_name):
            return finish({"ok": False, "error": "file_name must be a safe .txt basename"})

        topic, fields = TOPICS["outdoor"]
        measured = _read_measurement(config, topic, fields)
        if not measured.get("ok"):
            return finish({"ok": False, "error": measured.get("error", "MQTT read failed")})

        temperature = measured["temperature_c"]
        humidity = measured["humidity_percent"]
        line = (f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
                f"{temperature:.2f} C {humidity:.2f} %\n")
        payload = line.encode("utf-8")
        directory_fd = file_fd = None
        try:
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            file_fd = os.open(
                file_name,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o644, dir_fd=directory_fd,
            )
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise OSError("destination is not a regular file")
            written = os.write(file_fd, payload)
            if written != len(payload):
                raise OSError("short file write")
            os.fsync(file_fd)
            size = os.fstat(file_fd).st_size
        except OSError as exc:
            return finish({"ok": False, "error": f"file write failed: {exc}"})
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if directory_fd is not None:
                os.close(directory_fd)

        return finish({
            "ok": True,
            "file_path": str(root / file_name),
            "temperature_c": temperature,
            "humidity_percent": humidity,
            "line": line.strip(),
            "bytes_written": written,
            "file_size_bytes": size,
        })

    return record_outdoor_weather
