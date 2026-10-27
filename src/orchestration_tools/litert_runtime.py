"""Create the native manager using stock LiteRT-LM and our isolated tools.

Import LiteRT lazily: stage-1 IPC and stage-2 mock tests need no model installed.
"""
from __future__ import annotations

from pathlib import Path

from litert_agent_tools.config import MqttConfig
from litert_agent_tools.conversation import NativeConversationChat
from litert_agent_tools.mqtt_tools import create_mqtt_tools
from litert_agent_tools.shell_tools import ShellConfig, ShellGate, create_shell_tool

from .manager import NativeManager


def build_native_manager(
    model: Path, system_file: Path, *, enable_shell: bool = False,
    shell_cwd: Path = Path('/opt/model'), shell_timeout: float = 15.0,
    shell_max_output: int = 16384, enable_mqtt: bool = True,
    max_context_tokens: int = 4096, max_output_tokens: int = 512,
    cpu_threads: int | None = None, temperature: float = 0.2,
    top_p: float = 1.0, trace_native_json: bool = False,
) -> NativeManager:
    model = Path(model)
    system_file = Path(system_file)
    if not model.is_file():
        raise ValueError(f'Model not found: {model}')
    prompt = system_file.read_text(encoding='utf-8')
    if not prompt.strip():
        raise ValueError('Manager prompt is empty')
    if min(max_context_tokens, max_output_tokens) < 1:
        raise ValueError('Token limits must be positive')
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError('CPU threads must be positive')

    tools = create_mqtt_tools(MqttConfig.from_env()) if enable_mqtt else []
    shell_gate = None
    if enable_shell:
        shell_gate = ShellGate()
        tools.append(create_shell_tool(ShellConfig(
            shell_cwd, timeout_seconds=shell_timeout,
            max_output_bytes=shell_max_output,
        ), gate=shell_gate))

    import litert_lm

    engine = litert_lm.Engine(
        str(model), backend=litert_lm.Backend.CPU(thread_count=cpu_threads),
        vision_backend=litert_lm.Backend.CPU(),
        max_num_tokens=max_context_tokens,
        enable_ynnpack=False, enable_benchmark=True,
    )
    engine.__enter__()
    try:
        sampler = litert_lm.SamplerConfig(temperature=temperature, top_p=top_p, seed=0)
        chat = NativeConversationChat(
            engine, prompt, tools, sampler_config=sampler,
            max_output_tokens=max_output_tokens,
            thinking_config=litert_lm.ThinkingConfig(enable_thinking=False),
            trace_native_json=trace_native_json,
        )
        if shell_gate is not None:
            shell_gate.enabled = True
        return NativeManager(chat, engine, model_name=model.name)
    except BaseException:
        engine.close()
        raise
