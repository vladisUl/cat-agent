from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    cwd: Path
    operation: str
    metadata: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class CommandRuntime:
    """Execute a deliberately restricted Linux-like command set.

    Model output is never sent through a shell. Internal filesystem operations
    are confined to workspace. A tiny external-command allowlist is enabled by
    assigned skills and is executed with shell=False.
    """

    _SYSTEM_COMMANDS = {"date", "df", "free", "id", "nproc", "ps", "uname", "uptime", "whoami"}
    _MQTT_COMMANDS = {"mosquitto_pub", "mosquitto_sub"}

    def __init__(
        self,
        workspace: Path,
        skill_names: tuple[str, ...],
        *,
        max_file_bytes: int,
        timeout_seconds: int,
    ) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        self.root = workspace.resolve(strict=True)
        self.cwd = self.root
        self.skill_names = frozenset(skill_names)
        self.max_file_bytes = max_file_bytes
        self.timeout_seconds = timeout_seconds

    def prompt(self) -> str:
        return f"root@agent:{self.cwd}#"

    def execute(self, command: str) -> CommandResult:
        try:
            tokens = self._tokenize(command)
            if not tokens:
                return self._error(command, "unknown", 2, "bash: empty command", "empty_command")
            name = tokens[0]

            if "shell" in self.skill_names:
                internal = {
                    "pwd": self._pwd,
                    "cd": self._cd,
                    "ls": self._ls,
                    "cat": self._cat,
                    "sha256sum": self._sha256sum,
                    "test": self._test_file,
                    "[": self._test_file,
                    "mkdir": self._mkdir,
                    "touch": self._touch,
                    "cp": self._cp,
                    "mv": self._mv,
                    "rm": self._rm,
                    "echo": self._echo,
                }
                if name in internal:
                    return internal[name](command, tokens)
                if name in self._SYSTEM_COMMANDS:
                    return self._external(command, tokens)

            if "mqtt" in self.skill_names and name in self._MQTT_COMMANDS:
                return self._external(command, tokens)

            return self._error(
                command,
                name,
                126,
                f"bash: {name}: command is not permitted by assigned skills",
                "command_not_permitted",
            )
        except ValueError as exc:
            return self._error(command, "parse", 2, f"bash: parse error: {exc}", "parse_error")
        except OSError as exc:
            return self._error(
                command,
                "os",
                1,
                f"{exc.__class__.__name__}: {exc}",
                "os_error",
            )

    def format_result(self, result: CommandResult) -> str:
        parts = [f"root@agent:{result.cwd}# {result.command}"]
        if result.stdout:
            parts.extend(["--- stdout begin ---", result.stdout.rstrip("\n"), "--- stdout end ---"])
        if result.stderr:
            parts.extend(["--- stderr begin ---", result.stderr.rstrip("\n"), "--- stderr end ---"])
        parts.extend([f"[exit_code={result.exit_code}]", self.prompt()])
        return "\n".join(parts)

    def format_protocol_error(self, message: str) -> str:
        return f"protocol error: {message}\n{self.prompt()}"

    @staticmethod
    def _tokenize(command: str) -> list[str]:
        if any(char in command for char in ("\x00", "`", "$")):
            raise ValueError("command substitution and variable expansion are not permitted")
        tokens = shlex.split(command, posix=True)
        if any(char in command for char in (";", "|", "&", "<", "(" , ")")):
            raise ValueError("shell control operators are not permitted")
        if ">" in command and (not tokens or tokens[0] != "echo"):
            raise ValueError("redirection is permitted only for echo > FILE or echo >> FILE")
        return tokens

    def _pwd(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 1, "pwd")
        return self._ok(command, "pwd", f"{self.cwd}\n")

    def _cd(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 2, "cd PATH")
        target = self._existing(tokens[1])
        if not target.is_dir():
            return self._error(command, "cd", 1, f"cd: {tokens[1]}: Not a directory", "not_directory")
        old = self.cwd
        self.cwd = target
        return self._ok(command, "cd", metadata={"command_cwd": str(old), "path": str(target)})

    def _ls(self, command: str, tokens: list[str]) -> CommandResult:
        long_format = False
        hidden = False
        operands: list[str] = []
        for token in tokens[1:]:
            if token.startswith("-") and token != "-":
                flags = set(token[1:])
                if not flags.issubset({"l", "a"}):
                    return self._error(command, "ls", 2, f"ls: invalid option {token}", "invalid_option")
                long_format |= "l" in flags
                hidden |= "a" in flags
            else:
                operands.append(token)
        if len(operands) > 1:
            return self._error(command, "ls", 2, "ls: only one path is supported", "too_many_operands")
        display = operands[0] if operands else "."
        try:
            target = self._existing(display)
        except FileNotFoundError:
            return self._error(
                command, "ls", 2, f"ls: cannot access '{display}': No such file or directory", "not_found"
            )
        if target.is_file():
            return self._ok(command, "ls", self._ls_line(target, display, long_format) + "\n")
        entries = sorted(target.iterdir(), key=lambda item: item.name.casefold())
        if not hidden:
            entries = [item for item in entries if not item.name.startswith(".")]
        out = "\n".join(self._ls_line(item, item.name, long_format) for item in entries)
        return self._ok(command, "ls", out + ("\n" if out else ""))

    def _cat(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 2, "cat FILE")
        try:
            path = self._existing(tokens[1])
        except FileNotFoundError:
            return self._error(command, "cat", 1, f"cat: {tokens[1]}: No such file", "not_found")
        if not path.is_file():
            return self._error(command, "cat", 1, f"cat: {tokens[1]}: Not a regular file", "not_file")
        size = path.stat().st_size
        if size > self.max_file_bytes:
            return self._error(
                command,
                "cat",
                1,
                f"cat: file is {size} bytes; limit is {self.max_file_bytes}",
                "file_too_large",
            )
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace")
        return self._ok(
            command,
            "cat",
            text,
            metadata={"path": str(path), "size": size, "sha256": hashlib.sha256(data).hexdigest()},
        )

    def _sha256sum(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 2, "sha256sum FILE")
        try:
            path = self._existing(tokens[1])
        except FileNotFoundError:
            return self._error(command, "sha256sum", 1, "sha256sum: file not found", "not_found")
        if not path.is_file():
            return self._error(command, "sha256sum", 1, "sha256sum: not a regular file", "not_file")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return self._ok(command, "sha256sum", f"{digest}  {tokens[1]}\n")

    def _test_file(self, command: str, tokens: list[str]) -> CommandResult:
        normalized = tokens
        if tokens and tokens[0] == "[":
            if tokens[-1:] != ["]"]:
                return self._error(command, "test", 2, "[: missing ]", "parse_error")
            normalized = ["test", *tokens[1:-1]]
        if len(normalized) != 3 or normalized[1] not in {"-e", "-f", "-d"}:
            return self._error(command, "test", 2, "supported: test -e|-f|-d PATH", "invalid_test")
        try:
            path = self._candidate(normalized[2])
        except PermissionError as exc:
            return self._error(command, "test", 126, str(exc), "path_outside_workspace")
        flag = normalized[1]
        exists = path.exists() and (
            flag == "-e" or (flag == "-f" and path.is_file()) or (flag == "-d" and path.is_dir())
        )
        return CommandResult(command, 0 if exists else 1, "", "", self.cwd, "test", {"exists": exists})

    def _mkdir(self, command: str, tokens: list[str]) -> CommandResult:
        parents = False
        paths: list[str] = []
        for token in tokens[1:]:
            if token == "-p":
                parents = True
            elif token.startswith("-"):
                return self._error(command, "mkdir", 2, f"mkdir: unsupported option {token}", "invalid_option")
            else:
                paths.append(token)
        if len(paths) != 1:
            return self._error(command, "mkdir", 2, "mkdir supports one PATH", "invalid_arity")
        path = self._candidate(paths[0])
        path.mkdir(parents=parents, exist_ok=parents)
        return self._ok(command, "mkdir")

    def _touch(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 2, "touch FILE")
        path = self._candidate(tokens[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return self._ok(command, "touch")

    def _cp(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 3, "cp SOURCE DEST")
        source = self._existing(tokens[1])
        dest = self._candidate(tokens[2])
        if not source.is_file():
            return self._error(command, "cp", 1, "cp: source must be a regular file", "not_file")
        shutil.copy2(source, dest)
        return self._ok(command, "cp")

    def _mv(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 3, "mv SOURCE DEST")
        source = self._existing(tokens[1])
        dest = self._candidate(tokens[2])
        shutil.move(str(source), str(dest))
        return self._ok(command, "mv")

    def _rm(self, command: str, tokens: list[str]) -> CommandResult:
        self._arity(tokens, 2, "rm FILE")
        path = self._existing(tokens[1])
        if not path.is_file():
            return self._error(command, "rm", 1, "rm: only regular files are supported", "not_file")
        path.unlink()
        return self._ok(command, "rm")

    def _echo(self, command: str, tokens: list[str]) -> CommandResult:
        if ">" not in tokens and ">>" not in tokens:
            text = " ".join(tokens[1:]) + "\n"
            return self._ok(command, "echo", text)
        marker = ">>" if ">>" in tokens else ">"
        if tokens.count(marker) != 1:
            return self._error(command, "echo", 2, "echo: invalid redirection", "parse_error")
        index = tokens.index(marker)
        if index < 1 or index != len(tokens) - 2:
            return self._error(command, "echo", 2, "echo: use echo TEXT > FILE", "parse_error")
        text = " ".join(tokens[1:index]) + "\n"
        path = self._candidate(tokens[-1])
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if marker == ">>" else "w"
        with path.open(mode, encoding="utf-8") as handle:
            handle.write(text)
        return self._ok(command, "echo_write", metadata={"path": str(path), "bytes": len(text.encode())})

    def _external(self, command: str, tokens: list[str]) -> CommandResult:
        name = tokens[0]
        try:
            completed = subprocess.run(
                tokens,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return self._error(command, name, 127, f"bash: {name}: command not found", "command_not_found")
        except subprocess.TimeoutExpired:
            return self._error(command, name, 124, f"bash: {name}: command timed out", "timeout")
        return CommandResult(
            command,
            completed.returncode,
            completed.stdout,
            completed.stderr,
            self.cwd,
            name,
            {},
        )

    def _candidate(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.cwd / path
        resolved_parent = path.parent.resolve(strict=False)
        candidate = resolved_parent / path.name
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path outside workspace: {value}") from exc
        if candidate.is_symlink():
            raise PermissionError(f"symlink is not permitted: {value}")
        return candidate

    def _existing(self, value: str) -> Path:
        candidate = self._candidate(value)
        if not candidate.exists():
            raise FileNotFoundError(value)
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path outside workspace: {value}") from exc
        if candidate.is_symlink():
            raise PermissionError(f"symlink is not permitted: {value}")
        return resolved

    @staticmethod
    def _arity(tokens: list[str], expected: int, usage: str) -> None:
        if len(tokens) != expected:
            raise ValueError(f"usage: {usage}")

    @staticmethod
    def _ls_line(path: Path, display: str, long_format: bool) -> str:
        if not long_format:
            return display
        st = path.stat()
        mode = stat.filemode(st.st_mode)
        return f"{mode} {st.st_size:>10} {display}"

    def _ok(
        self,
        command: str,
        operation: str,
        stdout: str = "",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> CommandResult:
        return CommandResult(command, 0, stdout, "", self.cwd, operation, metadata or {})

    def _error(
        self,
        command: str,
        operation: str,
        exit_code: int,
        stderr: str,
        code: str,
    ) -> CommandResult:
        return CommandResult(command, exit_code, "", stderr, self.cwd, operation, {"error_code": code})
