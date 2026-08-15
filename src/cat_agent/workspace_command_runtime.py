from __future__ import annotations

import os
from pathlib import Path
import subprocess

from .command_runtime import CommandResult, CommandRuntime as RestrictedCommandRuntime


class CommandRuntime(RestrictedCommandRuntime):
    """Restricted command runtime plus direct execution of WORKSPACE programs."""

    def execute(self, command: str) -> CommandResult:
        try:
            tokens = self._tokenize(command)
        except ValueError:
            return super().execute(command)

        if (
            "shell" in self.skill_names
            and tokens
            and tokens[0].startswith("./")
        ):
            return self._workspace_executable(command, tokens)

        return super().execute(command)

    def _workspace_executable(
        self,
        command: str,
        tokens: list[str],
    ) -> CommandResult:
        display = tokens[0]
        try:
            program = self._existing(display)
        except FileNotFoundError:
            return self._error(
                command,
                "exec",
                127,
                f"bash: {display}: No such file or directory",
                "not_found",
            )
        except PermissionError as exc:
            return self._error(
                command,
                "exec",
                126,
                f"bash: {display}: {exc}",
                "path_outside_workspace",
            )

        if not program.is_file():
            return self._error(
                command,
                "exec",
                126,
                f"bash: {display}: not a regular file",
                "not_file",
            )
        if not os.access(program, os.X_OK):
            return self._error(
                command,
                "exec",
                126,
                f"bash: {display}: Permission denied",
                "not_executable",
            )

        argv = [str(program), *tokens[1:]]
        try:
            completed = subprocess.run(
                argv,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError:
            return self._error(
                command,
                "exec",
                127,
                f"bash: {display}: cannot execute: required interpreter not found",
                "interpreter_not_found",
            )
        except subprocess.TimeoutExpired:
            return self._error(
                command,
                "exec",
                124,
                f"bash: {display}: command timed out",
                "timeout",
            )
        except OSError as exc:
            return self._error(
                command,
                "exec",
                126,
                f"bash: {display}: {exc}",
                "exec_error",
            )

        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=self.cwd,
            operation="exec",
            metadata={"path": str(program)},
        )
