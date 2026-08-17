from __future__ import annotations

import subprocess

from .command_runtime import CommandResult, CommandRuntime as RestrictedCommandRuntime


class CommandRuntime(RestrictedCommandRuntime):
    """Use a real bash process when the agent has the shell skill.

    Commands still start in the configured workspace and keep the runtime timeout,
    but bash syntax itself is not restricted. Runtimes without the shell skill
    continue to use the restricted command implementation from command_runtime.py.
    """

    def execute(self, command: str) -> CommandResult:
        if "shell" in self.skill_names:
            return self._bash(command)
        return super().execute(command)

    def _bash(self, command: str) -> CommandResult:
        try:
            completed = subprocess.run(
                ["/bin/bash", "-c", command],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            if stderr and not stderr.endswith("\n"):
                stderr += "\n"
            stderr += "bash: command timed out"
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=stdout,
                stderr=stderr,
                cwd=self.cwd,
                operation="bash",
                metadata={"error_code": "timeout"},
            )
        except OSError as exc:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"bash: {exc}",
                cwd=self.cwd,
                operation="bash",
                metadata={"error_code": "exec_error"},
            )

        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=self.cwd,
            operation="bash",
            metadata={"shell": "/bin/bash"},
        )
