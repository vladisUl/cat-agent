from __future__ import annotations

import json
import re
import shlex
import subprocess

from .command_runtime import CommandResult, CommandRuntime as RestrictedCommandRuntime


WORK_PREFIX = "/work#"


def unwrap_work_command(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith(WORK_PREFIX):
        return None
    command = stripped[len(WORK_PREFIX):].strip()
    if not command or "\n" in command or "\r" in command:
        raise ValueError("command must be one non-empty line after /work#")
    return command


class CommandRuntime(RestrictedCommandRuntime):
    """Use a real bash process when the agent has the shell skill.

    Commands still start in the configured workspace and keep the runtime timeout,
    but bash syntax itself is not restricted. Runtimes without the shell skill
    continue to use the restricted command implementation from command_runtime.py.
    """

    _MQTT_HOST = "192.168.0.21"
    _MQTT_PORT = 1883
    _MQTT_FIELD_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def execute(self, command: str) -> CommandResult:
        stripped = command.strip()
        if "mqtt" in self.skill_names and (
            stripped == "mqtt_sub.sh" or stripped.startswith("mqtt_sub.sh ")
        ):
            return self._mqtt_sub_value(command)
        if "mqtt" in self.skill_names and (
            stripped == "mqtt_pub.sh" or stripped.startswith("mqtt_pub.sh ")
        ):
            return self._mqtt_pub_value(command)
        if "shell" in self.skill_names:
            return self._bash(command)
        return super().execute(command)

    def format_result(self, result: CommandResult) -> str:
        if result.operation == "mqtt_sub" and result.ok and result.stdout.strip():
            return result.stdout.strip()
        if result.operation == "mqtt_pub" and result.ok:
            return "OK"
        return super().format_result(result)

    def _mqtt_sub_value(self, command: str) -> CommandResult:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as exc:
            return self._error(command, "mqtt_sub", 2, f"mqtt_sub.sh: {exc}", "parse_error")

        if len(tokens) != 3 or tokens[0] != "mqtt_sub.sh":
            return self._error(
                command,
                "mqtt_sub",
                2,
                "mqtt_sub.sh: usage: mqtt_sub.sh TOPIC FIELD",
                "invalid_arguments",
            )

        topic, field = tokens[1], tokens[2]
        if not self._MQTT_FIELD_RE.fullmatch(field):
            return self._error(
                command,
                "mqtt_sub",
                2,
                "mqtt_sub.sh: FIELD must be a simple JSON field name",
                "invalid_field",
            )

        pipeline = (
            f"mosquitto_sub -h {self._MQTT_HOST} -p {self._MQTT_PORT} "
            f"-t {shlex.quote(topic)} -C 1 -W 5 | jq -r '.{field}'"
        )
        try:
            completed = subprocess.run(
                ["/bin/bash", "-o", "pipefail", "-c", pipeline],
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
            stderr += "mqtt_sub.sh: command timed out"
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=stdout,
                stderr=stderr,
                cwd=self.cwd,
                operation="mqtt_sub",
                metadata={"error_code": "timeout", "topic": topic, "field": field},
            )
        except OSError as exc:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"mqtt_sub.sh: {exc}",
                cwd=self.cwd,
                operation="mqtt_sub",
                metadata={"error_code": "exec_error", "topic": topic, "field": field},
            )

        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=self.cwd,
            operation="mqtt_sub",
            metadata={"topic": topic, "field": field},
        )

    def _mqtt_pub_value(self, command: str) -> CommandResult:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as exc:
            return self._error(command, "mqtt_pub", 2, f"mqtt_pub.sh: {exc}", "parse_error")

        if len(tokens) != 3 or tokens[0] != "mqtt_pub.sh":
            return self._error(
                command,
                "mqtt_pub",
                2,
                "mqtt_pub.sh: usage: mqtt_pub.sh TOPIC FIELD=VALUE",
                "invalid_arguments",
            )

        topic, assignment = tokens[1], tokens[2]
        if not topic or "+" in topic or "#" in topic:
            return self._error(
                command,
                "mqtt_pub",
                2,
                "mqtt_pub.sh: TOPIC must be a concrete MQTT topic",
                "invalid_topic",
            )
        if "=" not in assignment:
            return self._error(
                command,
                "mqtt_pub",
                2,
                "mqtt_pub.sh: FIELD=VALUE is required",
                "invalid_assignment",
            )

        field, raw_value = assignment.split("=", 1)
        if not self._MQTT_FIELD_RE.fullmatch(field):
            return self._error(
                command,
                "mqtt_pub",
                2,
                "mqtt_pub.sh: FIELD must be a simple JSON field name",
                "invalid_field",
            )

        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value

        publish_topic = topic if topic.endswith("/set") else f"{topic.rstrip('/')}/set"
        payload = json.dumps({field: value}, ensure_ascii=False, separators=(",", ":"))
        argv = [
            "mosquitto_pub",
            "-h",
            self._MQTT_HOST,
            "-p",
            str(self._MQTT_PORT),
            "-t",
            publish_topic,
            "-m",
            payload,
        ]
        try:
            completed = subprocess.run(
                argv,
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
            stderr += "mqtt_pub.sh: command timed out"
            return CommandResult(
                command=command,
                exit_code=124,
                stdout=stdout,
                stderr=stderr,
                cwd=self.cwd,
                operation="mqtt_pub",
                metadata={
                    "error_code": "timeout",
                    "topic": topic,
                    "publish_topic": publish_topic,
                    "field": field,
                    "value": value,
                },
            )
        except OSError as exc:
            return CommandResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=f"mqtt_pub.sh: {exc}",
                cwd=self.cwd,
                operation="mqtt_pub",
                metadata={
                    "error_code": "exec_error",
                    "topic": topic,
                    "publish_topic": publish_topic,
                    "field": field,
                    "value": value,
                },
            )

        return CommandResult(
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            cwd=self.cwd,
            operation="mqtt_pub",
            metadata={
                "topic": topic,
                "publish_topic": publish_topic,
                "field": field,
                "value": value,
            },
        )

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
