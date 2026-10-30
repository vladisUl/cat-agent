"""Create the native manager using stock LiteRT-LM and our isolated tools.

Import LiteRT lazily: stage-1 IPC and stage-2 mock tests need no model installed.
"""
from __future__ import annotations

from pathlib import Path
import logging

from litert_agent_tools.config import MqttConfig
from litert_agent_tools.conversation import NativeConversationChat
from litert_agent_tools.mqtt_tools import BUILTIN_MQTT_TOOL_NAMES, create_mqtt_tools
from litert_agent_tools.shell_tools import ShellConfig, ShellGate, create_shell_tool

from .agent_config import effective_agent_tool_names, load_agent_specs
from .agents import AgentPool, NativeAgent
from .delegation import DelegationBroker, create_delegate_tool
from .manager import NativeManager
from .task_tools import NativeTaskController, create_task_tools

LOGGER = logging.getLogger(__name__)


def build_native_manager(
    model: Path, system_file: Path, *, enable_shell: bool = False,
    shell_cwd: Path = Path('/opt/model'), shell_timeout: float = 15.0,
    shell_max_output: int = 16384, enable_mqtt: bool = True,
    max_context_tokens: int = 4096, max_output_tokens: int = 512,
    cpu_threads: int | None = None, backend_name: str = "cpu",
    vision_backend_name: str = "cpu",
    activation_dtype: str | None = None, speculative: bool = False,
    temperature: float = 0.2,
    top_p: float = 1.0, trace_native_json: bool = False,
    agent_config: Path | None = None,
    task_socket: Path | None = None,
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
    if backend_name not in ('cpu', 'gpu'):
        raise ValueError(f'Native LiteRT backend must be cpu or gpu: {backend_name!r}')
    if vision_backend_name not in ('cpu', 'gpu', 'none'):
        raise ValueError(f'Native LiteRT vision backend must be cpu, gpu or none: {vision_backend_name!r}')
    if activation_dtype is not None and activation_dtype not in ('fp32', 'fp16', 'int16', 'int8'):
        raise ValueError(f'Unsupported LiteRT activation dtype: {activation_dtype!r}')

    # AGENT prompts live alongside the external manager prompt.
    # Relative AGENT paths are resolved against the manager prompt's directory.
    prompt_dir = system_file.parent
    specs = (
        load_agent_specs(agent_config, prompt_dir=prompt_dir)
        if agent_config is not None else ()
    )
    if specs:
        prompt += (
            "\n\nДоступные независимые исполнители AGENT: "
            + ", ".join(spec.agent_id for spec in specs)
            + ". Для передачи задания используй инструмент delegate_task."
        )
    # Read-only MQTT functions are built-in capabilities of both MANAGER
    # and every native AGENT. Shell is privileged and remains opt-in for
    # each AGENT through its explicit JSON tool grant.
    mqtt_tools = create_mqtt_tools(MqttConfig.from_env()) if enable_mqtt else []
    mqtt_names = tuple(fn.__name__ for fn in mqtt_tools)
    if enable_mqtt and mqtt_names != BUILTIN_MQTT_TOOL_NAMES:
        raise RuntimeError("MQTT tool registry does not match the native TASK skill registry")
    tools = list(mqtt_tools)
    shell_gate = None
    if enable_shell:
        shell_gate = ShellGate()
        tools.append(create_shell_tool(ShellConfig(
            shell_cwd, timeout_seconds=shell_timeout,
            max_output_bytes=shell_max_output,
        ), gate=shell_gate))

    available_tools = {fn.__name__: fn for fn in tools}
    for spec in specs:
        missing = set(spec.tool_names) - set(available_tools)
        if missing:
            raise ValueError(
                f"{spec.agent_id}: unavailable or disabled tools: {sorted(missing)}"
            )
        if not spec.prompt_file.read_text(encoding="utf-8").strip():
            raise ValueError(f"{spec.agent_id}: agent prompt is empty")

    # Manager-only tool. Do not place it in available_tools: AGENTs cannot
    # delegate, and must not be able to request nested model inference.
    delegations = DelegationBroker(tuple(spec.agent_id for spec in specs)) if specs else None
    task_controller = None
    if task_socket is not None:
        if not specs:
            raise ValueError("Native periodic tasks require configured AGENTs")
        from task_system.client import TaskRPC
        permissions = {
            spec.agent_id: effective_agent_tool_names(spec, mqtt_names)
            for spec in specs
        }
        task_controller = NativeTaskController(TaskRPC(task_socket), permissions)
        tools.extend(create_task_tools(task_controller))
        prompt += (
            "\n\nПоручения AGENT ставь через delegate_task(agent_id, task, period_seconds). "
            "Если нужно выполнить один раз, period_seconds=0. "
            "Если повторять каждую минуту, period_seconds=60. "
            "Передавай в task только содержание работы, БЕЗ временных условий: "
            "не включай 'каждую минуту', 'через час' и другие интервалы. "
            "Время передавай ТОЛЬКО отдельным полем period_seconds. "
            "Тактику выполнения и необходимость ответа определяет AGENT. "
            "Если сообщение не требуется, AGENT может закончить молча. "
            "Повторения организует TASK SYSTEM, не shell или AGENT. "
            "Для возобновления остановленного задания используй start_periodic_query(task_id), "
            "для остановки (с сохранением TASK) — stop_periodic_query(task_id), "
            "для полного удаления сохранённого задания — delete_periodic_query(task_id), "
            "для списка — list_periodic_queries(). "
            "Не объявляй периодическое задание созданным без ok=true и task_id."
        )
    if delegations is not None:
        delegations.periodic_controller = task_controller
        tools.append(create_delegate_tool(delegations))
    LOGGER.info("native-tools MANAGER registered=%s",
                tuple(fn.__name__ for fn in tools))

    import litert_lm

    model_backend = (
        litert_lm.Backend.GPU() if backend_name == 'gpu'
        else litert_lm.Backend.CPU(thread_count=cpu_threads)
    )
    engine_options = dict(
        backend=model_backend,
        max_num_tokens=max_context_tokens,
        enable_speculative_decoding=speculative,
        enable_benchmark=True,
    )
    # YNNPACK belongs to CpuConfig only. Passing it to GpuArtisanConfig
    # produces INVALID_ARGUMENT even with enable_ynnpack=False.
    if backend_name == 'cpu':
        # The tested CPU LiteRT path failed with YNNPACK UNIMPLEMENTED.
        engine_options['enable_ynnpack'] = False
    # LiteRT 0.18.x tries to load tf_lite_vision_encoder whenever
    # vision_backend is supplied. Text-only E2B GPU models lack that section.
    # Omitting the argument disables vision executor creation.
    if vision_backend_name == 'cpu':
        engine_options['vision_backend'] = litert_lm.Backend.CPU()
    elif vision_backend_name == 'gpu':
        engine_options['vision_backend'] = litert_lm.Backend.GPU()
    if activation_dtype is not None:
        activation = litert_lm.ActivationDataType.from_str(activation_dtype)
        if activation is None:
            raise ValueError(f'LiteRT does not support activation dtype {activation_dtype!r}')
        engine_options['activation_data_type'] = activation
    LOGGER.info('Native LiteRT backend=%s vision_backend=%s speculative=%s activation_dtype=%s',
                backend_name, vision_backend_name, speculative, activation_dtype or 'default')
    engine = litert_lm.Engine(str(model), **engine_options)
    engine.__enter__()
    chat = None
    agents = []
    try:
        sampler = litert_lm.SamplerConfig(temperature=temperature, top_p=top_p, seed=0)
        thinking = litert_lm.ThinkingConfig(enable_thinking=False)
        chat = NativeConversationChat(
            engine, prompt, tools, sampler_config=sampler,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking,
            trace_native_json=trace_native_json,
        )
        for spec in specs:
            # Default MQTT tools require no per-agent JSON repetition.
            # Explicit grants (for example shell) extend that safe baseline,
            # without registering the same tool twice.
            agent_tool_names = effective_agent_tool_names(spec, mqtt_names)
            agent_tools = [available_tools[name] for name in agent_tool_names]
            LOGGER.info("native-tools %s registered=%s", spec.agent_id, agent_tool_names)
            agent_chat = NativeConversationChat(
                engine,
                spec.prompt_file.read_text(encoding="utf-8"),
                agent_tools,
                sampler_config=sampler,
                max_output_tokens=max_output_tokens,
                thinking_config=thinking,
                trace_native_json=trace_native_json,
            )
            agents.append(NativeAgent(spec.agent_id, agent_chat))
        if shell_gate is not None:
            shell_gate.enabled = True
        if delegations is not None:
            delegations.enabled = True
        if task_controller is not None:
            task_controller.enabled = True
        return NativeManager(
            chat, engine, model_name=model.name, backend_name=backend_name,
            agents=AgentPool(agents), delegations=delegations,
            task_controller=task_controller,
        )
    except BaseException:
        for agent in reversed(agents):
            agent.close()
        if chat is not None:
            chat.close()
        engine.close()
        raise
