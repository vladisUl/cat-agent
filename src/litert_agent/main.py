from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import litert_lm

from orchestration.config import Settings
from orchestration.mqtt_events import MqttEventMonitor

from .core_scheduler import HARDWARE_EVENT_PRIORITY
from .core_server import CoreServer
from .runtime import build_bundle, warm_bundle

LOGGER = logging.getLogger(__name__)
LOG_DIR = Path("/var/log/litertlm")
LOG_PATH = LOG_DIR / "cat-agent.log"


class _ProtocolLogFilter(logging.Filter):
    """Turn noisy orchestration INFO into the actual protocol dialogue.

    This filter changes logging presentation only. It does not participate in
    routing, scheduling, model calls, task execution or tool execution.
    Warnings and errors are always preserved verbatim.
    """

    _ORCHESTRATION_LOGGERS = {
        "orchestration.assistant_manager",
        "orchestration.manager",
        "orchestration.agent",
        "orchestration.system_events",
        "litert_agent.core_scheduler",
    }

    @staticmethod
    def _args(record: logging.LogRecord) -> tuple[object, ...]:
        return record.args if isinstance(record.args, tuple) else ()

    @staticmethod
    def _replace(record: logging.LogRecord, message: str, *args: object) -> bool:
        record.msg = message
        record.args = args
        return True

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno <= logging.DEBUG:
            return True
        if record.name not in self._ORCHESTRATION_LOGGERS:
            return True

        message = str(record.msg)
        args = self._args(record)

        try:
            if message.startswith("MANAGER USER MESSAGE") and args:
                return self._replace(record, "USER -> MANAGER %r", args[-1])

            if "MODEL RESPONSE\n%s" in message and args:
                source = "MANAGER" if "manager" in message.lower() else str(args[0]).upper()
                text = str(args[-1]).strip()
                if source == "MANAGER":
                    destination = "USER" if text.startswith(("REPLY", "ASK")) else "SYSTEM"
                else:
                    destination = "SYSTEM"
                return self._replace(
                    record,
                    f"{source} -> {destination} %r",
                    text,
                )

            if message == "MANAGER WORK RESULT\n%s" and args:
                return self._replace(record, "SYSTEM -> MANAGER %r", args[-1])

            if message.startswith("MANAGER autonomous QUERY tick") and args:
                return self._replace(record, "SYSTEM -> MANAGER %r", args[-1])

            if message.startswith("MANAGER SYSTEM EVENT") and args:
                return self._replace(record, "SYSTEM -> MANAGER %r", args[-1])

            if " START method=%s skills=%s bootstrap_reused=%s workspace=%s task=%r" in message:
                if len(args) >= 6:
                    agent_id = str(args[0]).upper()
                    method = str(args[1])
                    task = str(args[5])
                    payload: dict[str, str] = {}
                    if method != "ordinary":
                        payload["method"] = method
                    payload["task"] = task
                    actual_input = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                    return self._replace(
                        record,
                        f"SYSTEM -> {agent_id} %r",
                        actual_input,
                    )

            if " TOOL RESULT operation=%s exit=%d metadata=%r\n%s" in message:
                if len(args) >= 6:
                    agent_id = str(args[0]).upper()
                    return self._replace(
                        record,
                        f"SYSTEM -> {agent_id} %r",
                        args[-1],
                    )

            if message == "%s step %d DEFERRED TOOL RESULT command=%s\n%s":
                if len(args) >= 4:
                    agent_id = str(args[0]).upper()
                    return self._replace(
                        record,
                        f"SYSTEM -> {agent_id} %r",
                        args[-1],
                    )

            if message == "%s CONTINUE context=%r" and len(args) >= 2:
                agent_id = str(args[0]).upper()
                actual_input = json.dumps(
                    {"context": str(args[1]).strip()},
                    ensure_ascii=False,
                    indent=2,
                ) + "\n"
                return self._replace(
                    record,
                    f"SYSTEM -> {agent_id} %r",
                    actual_input,
                )

            if message.startswith("MANAGER DIRECT TOOL RESULT") and args:
                return self._replace(record, "SYSTEM -> MANAGER %r", args[-1])
        except Exception:
            # Logging must never be able to affect the running agent.
            return True

        # All other INFO from these modules is technical scheduler/runtime
        # bookkeeping. Keep it available at DEBUG instead of mixing it with the
        # protocol transcript shown at normal INFO level.
        return False


def _enqueue_mqtt_event(core: CoreServer, bundle, binding, value: str) -> None:
    event = bundle.runtime.external_event("mqtt", binding.name, value=value)
    if event is None:
        return
    core.scheduler.enqueue_external_event(
        event,
        priority=HARDWARE_EVENT_PRIORITY,
        coalesce=False,
    )


def main() -> int:
    settings = Settings.from_env(require_model=False)
    try:
        _configure_logging(settings)
    except OSError as exc:
        print(f"Cannot initialize log {LOG_PATH}: {exc}", file=sys.stderr)
        return 2

    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    LOGGER.info("cat-agent CORE starting")
    LOGGER.info("Log file: %s", LOG_PATH)
    LOGGER.info("Workspace: %s", settings.workspace)
    LOGGER.info("Prompt dir: %s", settings.prompt_dir)

    bundle = build_bundle(settings)
    try:
        manager_warm, agent_warm = warm_bundle(bundle, settings)
        LOGGER.info(
            "LiteRT prefixes ready: manager=%d tokens %.3fs, agent=%d tokens %.3fs",
            manager_warm.token_count,
            manager_warm.elapsed_seconds,
            agent_warm.token_count,
            agent_warm.elapsed_seconds,
        )

        bundle.system_runtime.arm_task_timers()
        LOGGER.info("SYSTEM persistent task timers armed after model warmup")

        core = CoreServer(bundle)
        mqtt_monitor = MqttEventMonitor(
            bundle.runtime.event_store,
            lambda binding, value: _enqueue_mqtt_event(core, bundle, binding, value),
        )
        core.start()
        mqtt_monitor.start()
        try:
            core.serve_forever()
        finally:
            mqtt_monitor.close()
            core.close()
        return 0
    finally:
        bundle.close()
        LOGGER.info("cat-agent CORE stopped")


def _configure_logging(settings: Settings) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    level = getattr(logging, settings.log_level, logging.INFO)
    protocol_filter = _ProtocolLogFilter()

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(protocol_filter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(protocol_filter)
    root.addHandler(console_handler)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
