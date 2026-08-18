from __future__ import annotations

import logging

from .manager import ManagerRuntime, ManagerTurn
from .model_client import ModelClientError, OpenAIChatClient

LOGGER = logging.getLogger(__name__)


class DirectSessionManagerRuntime(ManagerRuntime):
    """Manager with a separate resident direct session for the САМ prefix."""

    def __init__(self, client: OpenAIChatClient, direct_client: OpenAIChatClient, *args, **kwargs) -> None:
        super().__init__(client, *args, **kwargs)
        self.direct_client = direct_client
        direct_system_context = self.prompt_store.build_direct_system_context(
            self._manager_skills, self._manager_workspace
        )
        self._direct_base_messages = [{"role": "system", "content": direct_system_context}]
        self._direct_messages = [dict(item) for item in self._direct_base_messages]

    @property
    def direct_base_messages(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._direct_base_messages]

    def user_message(self, text: str) -> ManagerTurn:
        user_text = text.strip()
        folded = user_text.casefold()
        if folded == "конец чата":
            self._abort_direct()
            return super().user_message(text)

        if self._direct_waiting:
            model_text = user_text
            self._direct_waiting = False
            LOGGER.info("MANAGER DIRECT CONTINUE raw=%r model=%r", user_text, model_text)
            self._append_direct_user(model_text)
            return self._drive_direct_session()

        parts = user_text.split(maxsplit=1)
        if parts and parts[0].casefold() == "сам":
            task_text = parts[1].strip() if len(parts) == 2 else ""
            model_text = task_text
            self._direct_mode = True
            self._direct_waiting = False
            self._direct_repeated = {}
            self._reset_direct_to_base()
            LOGGER.info(
                "MANAGER USER MESSAGE chat=%s direct=True raw=%r model=%r",
                self._chat_mode,
                user_text,
                model_text,
            )
            self._append_direct_user(model_text)
            return self._drive_direct_session()

        return super().user_message(text)

    def _drive_direct_session(self) -> ManagerTurn:
        for step in range(1, self.max_steps + 1):
            if not self._direct_messages or self._direct_messages[-1]["role"] != "user":
                self._abort_direct()
                return ManagerTurn("error", "TT violation: manager direct model call requires a fresh user tick")

            try:
                response = self.direct_client.chat(self._direct_messages)
            except ModelClientError as exc:
                LOGGER.exception("manager direct step %d model request failed", step)
                self._abort_direct()
                return ManagerTurn("error", f"Model request failed: {exc}")

            LOGGER.info(
                "manager direct step %d response in %.3f s: prompt=%s cached=%s new=%s prefill=%s completion=%s generate=%s",
                step,
                response.elapsed_seconds,
                response.prompt_tokens if response.prompt_tokens is not None else "?",
                response.cached_tokens if response.cached_tokens is not None else "?",
                response.prompt_evaluated_tokens if response.prompt_evaluated_tokens is not None else "?",
                f"{response.prompt_seconds:.3f}s" if response.prompt_seconds is not None else "?",
                response.completion_tokens if response.completion_tokens is not None else "?",
                f"{response.generation_seconds:.3f}s" if response.generation_seconds is not None else "?",
            )
            LOGGER.info("manager direct step %d MODEL RESPONSE\n%s", step, response.content)
            self._direct_messages.append({"role": "assistant", "content": response.content})

            output = response.content.strip()

            if output == "ASK" or output.startswith("ASK\n"):
                body = output[3:].strip()
                self._direct_waiting = True
                LOGGER.info("MANAGER DIRECT NEED steps=%d text=%r", step, body)
                return ManagerTurn("ask", body)

            if output.startswith("/work#"):
                command = output[len("/work#"):].strip()
                if not command or "\n" in command or "\r" in command:
                    message = "direct command must be one non-empty line after /work#"
                    LOGGER.warning("manager direct step %d protocol error: %s", step, message)
                    self._append_direct_user(self._direct_runtime.format_protocol_error(message))
                    continue

                LOGGER.info("MANAGER DIRECT TOOL COMMAND %s", command)
                result = self._direct_runtime.execute(command)
                formatted = self._direct_runtime.format_result(result)
                LOGGER.info(
                    "MANAGER DIRECT TOOL RESULT operation=%s exit=%d metadata=%r\n%s",
                    result.operation,
                    result.exit_code,
                    result.metadata,
                    formatted,
                )
                signature = (command, result.exit_code, result.stdout, result.stderr)
                self._direct_repeated[signature] = self._direct_repeated.get(signature, 0) + 1
                if self._direct_repeated[signature] > 2:
                    LOGGER.error("MANAGER DIRECT FAILED same command/result repeated more than twice: %s", command)
                    self._abort_direct()
                    return ManagerTurn("error", "same command with the same result repeated more than twice")
                self._append_direct_user(formatted)
                continue

            if output.startswith("REPLY\n"):
                output = output[len("REPLY\n"):].strip()

            LOGGER.info("MANAGER DIRECT COMPLETE steps=%d result=%r", step, output)
            self._direct_mode = False
            self._direct_waiting = False
            self._direct_repeated = {}
            self._reset_direct_to_base()
            return ManagerTurn("reply", output)

        LOGGER.error("Manager direct execution exceeded maximum of %d steps", self.max_steps)
        self._abort_direct()
        return ManagerTurn("error", f"Manager direct execution exceeded maximum of {self.max_steps} steps")

    def _append_direct_user(self, text: str) -> None:
        content = text.strip()
        if self._direct_messages and self._direct_messages[-1]["role"] == "user":
            previous = self._direct_messages[-1]["content"].rstrip()
            self._direct_messages[-1]["content"] = f"{previous}\n\n{content}" if previous else content
            return
        self._direct_messages.append({"role": "user", "content": content})

    def _abort_direct(self) -> None:
        self._direct_mode = False
        self._direct_waiting = False
        self._direct_repeated = {}
        self._reset_direct_to_base()

    def _reset_direct_to_base(self) -> None:
        try:
            reset_to_base = getattr(self.direct_client, "reset_to_base", None)
            if callable(reset_to_base):
                reset_to_base(self._direct_base_messages)
            LOGGER.info("MANAGER DIRECT RESET resident session to base")
        except Exception:
            LOGGER.debug("MANAGER DIRECT resident session is not prepared yet; local history reset only", exc_info=True)
        self._direct_messages = [dict(item) for item in self._direct_base_messages]
