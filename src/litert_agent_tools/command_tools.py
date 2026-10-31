"""Native LiteRT command tools: JSON definitions -> typed Python callables."""
from __future__ import annotations

import json
import keyword
import logging
import math
import os
from pathlib import Path
import re
import selectors
import subprocess
import time
from typing import Callable

from .shell_tools import ShellGate, _kill_process_group

LOGGER = logging.getLogger(__name__)
CONFIG_FILE = Path(__file__).resolve().parents[2] / "config/command_tools.json"
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
EXECUTABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
KINDS = {"string": str, "integer": int, "number": float, "boolean": bool}
MAX_ARG_LENGTH = 1024


def _definitions(path: Path) -> list[tuple]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load command tools JSON {path}: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"tools"} or not isinstance(data["tools"], list):
        raise ValueError(f"{path}: expected a tools array")
    result = []
    used = set()
    for index, item in enumerate(data["tools"]):
        label = f"{path} tools[{index}]"
        if not isinstance(item, dict) or not (
            {"name", "description", "command"} <= set(item)
            <= {"name", "description", "command", "parameters"}
        ):
            raise ValueError(f"{label}: expected name/description/command and optional parameters")
        name, description, command = item["name"], item["description"], item["command"]
        if (not isinstance(name, str) or not NAME.fullmatch(name)
                or keyword.iskeyword(name) or name in used):
            raise ValueError(f"{label}: invalid or duplicate tool name {name!r}")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"{label}: description must be nonempty")
        if not isinstance(command, str) or not EXECUTABLE.fullmatch(command):
            raise ValueError(f"{label}: command must be one executable basename")
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"{label}: parameters must be an object")
        parsed = []
        for key, spec in parameters.items():
            if (not isinstance(key, str) or not NAME.fullmatch(key)
                    or keyword.iskeyword(key)):
                raise ValueError(f"{label}: invalid parameter name {key!r}")
            if (not isinstance(spec, dict) or
                    not {"type", "description"} <= set(spec) <= {"type", "description", "enum"} or
                    not isinstance(spec.get("type"), str) or spec["type"] not in KINDS or
                    not isinstance(spec.get("description"), str) or
                    not spec["description"].strip()):
                raise ValueError(f"{label}: invalid parameter specification for {key}")
            choices = spec.get("enum")
            if choices is not None:
                if (not isinstance(choices, list) or not choices or
                        any(not _valid_type(v, spec["type"]) for v in choices) or
                        len(set(map(repr, choices))) != len(choices)):
                    raise ValueError(f"{label}: invalid enum for {key}")
            parsed.append((key, spec["type"], spec["description"], choices))
        result.append((name, description, command, tuple(parsed)))
        used.add(name)
    return result


def _valid_type(value, kind: str) -> bool:
    if kind == "number":
        return type(value) in (int, float) and math.isfinite(value)
    return type(value) is KINDS[kind]


def _run(executable: Path, directory: Path, arguments: list[str],
         timeout: float, max_output_bytes: int) -> dict:
    process = None
    out, err = bytearray(), bytearray()
    error = None
    exit_code = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            [str(executable), *arguments],
            cwd=directory, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True,
        )
        assert process.stdout is not None and process.stderr is not None
        deadline = started + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, out)
            selector.register(process.stderr, selectors.EVENT_READ, err)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    error = "timeout"
                    break
                ready = selector.select(remaining)
                if not ready:
                    continue
                for key, _ in ready:
                    budget = max_output_bytes - len(out) - len(err)
                    chunk = os.read(key.fileobj.fileno(), min(8192, budget + 1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    key.data.extend(chunk[:budget])
                    if len(chunk) > budget:
                        error = "output_limit_exceeded"
                        break
                if error is not None:
                    break
        if error is None:
            try:
                exit_code = process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                error = "timeout"
    except OSError as exc:
        error = f"process_error: {exc}"
    finally:
        if process is not None:
            if error is not None or process.poll() is None:
                _kill_process_group(process)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()

    stdout = out.decode("utf-8", errors="replace").strip()
    stderr = err.decode("utf-8", errors="replace").strip()
    if error is not None:
        return {"ok": False, "exit_code": exit_code, "error": error}
    if exit_code != 0:
        return {"ok": False, "exit_code": exit_code,
                "error": stderr[:500] or f"exit_code_{exit_code}"}
    try:
        result = json.loads(stdout)
    except ValueError:
        result = stdout
    return {"ok": True, "result": result}


def _make_tool(name: str, description: str, executable: Path, directory: Path,
               parameters: tuple, gate: ShellGate | None, timeout: float,
               max_output_bytes: int) -> Callable:
    def invoke(*values) -> dict:
        if gate is not None and not gate.enabled:
            return {"ok": False, "error": "command tools disabled during BASE warm-up"}
        args = []
        for value, (key, kind, _, choices) in zip(values, parameters):
            if not _valid_type(value, kind):
                return {"ok": False, "error": f"{key}: expected {kind}"}
            if choices is not None and value not in choices:
                return {"ok": False, "error": f"{key}: value outside enum"}
            argument = str(value).lower() if kind == "boolean" else str(value)
            if len(argument) > MAX_ARG_LENGTH or "\x00" in argument:
                return {"ok": False, "error": f"{key}: invalid argument length"}
            args.append(argument)
        LOGGER.info("native-tool CALL name=%s args=%r", name, args)
        outcome = _run(executable, directory, args, timeout, max_output_bytes)
        LOGGER.info("native-tool RESULT name=%s ok=%s error=%s",
                    name, outcome["ok"], outcome.get("error"))
        return outcome

    # Parameter names are validated Python identifiers; only fixed source
    # fragments are compiled. The model sees a real typed function signature.
    signature = ", ".join(f"{key}: {kind.replace('integer', 'int').replace('number', 'float').replace('boolean', 'bool').replace('string', 'str')}"
                          for key, kind, _, _ in parameters)
    arguments = ", ".join(key for key, _, _, _ in parameters)
    scope = {"_invoke": invoke, "__name__": __name__}
    exec(compile(f"def {name}({signature}) -> dict:\n    return _invoke({arguments})\n",
                 "<native-command-tools>", "exec"), scope)
    tool = scope[name]
    lines = [description]
    if parameters:
        lines += ["", "Args:"]
        for key, kind, explanation, choices in parameters:
            detail = explanation
            if choices is not None:
                detail += " Допустимые значения: " + ", ".join(map(str, choices)) + "."
            lines.append(f"    {key} ({kind}): {detail}")
    tool.__doc__ = "\n".join(lines)
    return tool


def create_command_tools(
    command_dir: Path, config_file: Path | None = None, *, gate: ShellGate | None = None,
    timeout_seconds: float = 15.0, max_output_bytes: int = 16384,
) -> list[Callable]:
    """Load configured executable tools as regular functions for LiteRT Conversation."""
    if not 0 < timeout_seconds <= 120:
        raise ValueError("Command tool timeout must be in (0, 120]")
    if not 1 <= max_output_bytes <= 65536:
        raise ValueError("Command tool output limit must be in 1..65536")
    directory = Path(command_dir)
    if not directory.is_absolute():
        raise ValueError("Command tool directory must be absolute")
    definitions = _definitions(Path(config_file) if config_file else CONFIG_FILE)
    if definitions and not directory.is_dir():
        raise ValueError(f"Command tool directory not found: {directory}")
    directory = directory.resolve()
    tools = []
    for name, description, command, parameters in definitions:
        executable = (directory / command).resolve()
        if (executable.parent != directory or not executable.is_file()
                or not os.access(executable, os.X_OK)):
            raise ValueError(f"{name}: executable is missing or outside command directory: {command}")
        tools.append(_make_tool(name, description, executable, directory, parameters,
                                gate, timeout_seconds, max_output_bytes))
    LOGGER.info("native-tools COMMAND loaded=%s from=%s",
                tuple(tool.__name__ for tool in tools), directory)
    return tools
