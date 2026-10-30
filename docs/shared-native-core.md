# Новый native LiteRT CORE и общий TASK SYSTEM

## Что переделано (октябрь 2026)

Для `cat-agent` создан **второй, независимый LiteRT CORE** с нативным
`Conversation` / Tool Calling вместо текстовых команд `/work#`.
Старые OpenAI CORE и LiteRT CORE не заменены; OpenAI продолжает работать
с облачными моделями через OpenAI-compatible API.

- `litert_agent_tools` — LiteRT Engine, прогретый BASE, независимые
  Conversation/KV и Python-инструменты, вызываемые моделью нативно.
- `orchestration_tools` — MANAGER, отдельные AGENT с собственными промптами,
  BASE и наборами разрешённых инструментов; `delegate_task` не запускает
  вложенный инференс прямо из tool callback.
- `agent_core_tools` — отдельный Unix IPC, один последовательный исполнитель
  модели, приоритетный scheduler (control / human / hardware / background)
  и обработчик заданий общего TASK SYSTEM.
- `TASK SYSTEM` — **один процесс и одна SQLite** для TASK, RUN, таймеров,
  MQTT-событий, готовности CORE и доставляемых QUERY-результатов.
  Native worker зарегистрирован как `executor=litert`, облачный —
  как `executor=openai`.
- `start_tui_tools.sh` — отдельный TUI для native CORE. QUERY-уведомления
  сохраняются в общей базе, показываются при подключении и подтверждаются
  идемпотентным ACK через native CORE.

Старые `tools/*.md`, `/work#` и старые агенты относятся к **прежнему CORE**.
Не следует считать их протоколом нового native CORE.

## Рабочая схема и запуск

```text
./start_tui.sh openai  -> /run/cat-agent/openai.sock
                         OpenAI CORE (Ollama Cloud /v1)
                                     \
                                      \ executor=openai
                                       \
                     /run/cat-agent/tasks.sock
                        один TASK SYSTEM + SQLite
                                       /
                                      / executor=litert
./start_tui_tools.sh -> /run/cat-agent/litert-tools.sock
                         native LiteRT CORE (CPU/GPU, native tools)
```

Из `/opt/cat-agent` запустить в отдельных терминалах, **если процессы
ещё не запущены**:

```bash
./start_task_system.sh
./start_openai_agent.sh
./start_native_core.sh
```

Подключение TUI:

```bash
./start_tui.sh openai
./start_tui_tools.sh
```

Проверка общего состояния и очереди:

```bash
./task_system.sh cores
./task_system.sh list
./task_system.sh runs
```

`./start_tui.sh litert` использует **старый** LiteRT IPC на `litert.sock`
и не может подключиться к новому native CORE. Для нового CORE используйте
`./start_tui_tools.sh`.

После перехода на native CORE команда запуска одна:
`./start_native_core.sh`. Промежуточная обёртка
`start_litert_agent_tools.sh` и прежний `start_litert_agent.sh`
удалены. Внутри launcher напрямую исполняется
`/opt/litert-lm-venv/bin/python3 -m agent_core_tools`,
с сохранением `CAT_AGENT_TOOLS_PYTHON` для отладки и тестов.

`systemd/cat-agent-litert.service` также запускает
`start_native_core.sh`. Его тип — `simple`, а не `notify`,
поскольку native CORE не отправляет `sd_notify(READY=1)`.
При установке systemd-unit вне репозитория его надо обновить
отдельно; `git pull` не перезаписывает `/etc/systemd/system`.
Прежние TUI/Web/Voice для `litert.sock` пока не поддерживают
протокол нового CORE, поэтому для диалога используйте
`start_tui_tools.sh`.

`start_native_core.sh` берёт активный CPU/GPU-профиль из `cat-agent.yaml`
(`litert.active_profile`), `prompts_tools/manager.md`, 
`config/native_agents.example.json` и каталог работы `runtime.workspace`.
Для другой модели можно указать `CAT_AGENT_NATIVE_MODEL_PATH`, для другого
Python — `CAT_AGENT_TOOLS_PYTHON`. Рабочее Python-окружение WSL:
`/opt/litert-lm-venv/bin/python3`.

## Режим готовности облачного CORE

Текущий профиль `openai.profiles.cloud` использует:

```yaml
base_url: http://127.0.0.1:11434/v1
model: gemma4:31b-cloud
readiness: openai
```

`readiness: openai` проверяет `/v1/models` и наличие модели,
**но не проверяет квоту**. `readiness: ollama` дополнительно проверяет
`/api/tags` и облачную квоту через недокументированный `/api/usage`;
изменение его схемы может дать `usage_invalid_response`.
`executor=auto` предпочитает OpenAI при READY, затем LiteRT. Поэтому при
`readiness: openai` исчерпание облачной квоты само по себе **не**
переведёт задания на LiteRT.

## Нынешние границы / недоделки

- Native CORE выбирает **CPU или GPU** из `litert.active_profile`.
  GPU и speculative передаются в LiteRT Engine согласно YAML; реальная
  работоспособность GPU, в том числе `Conversation.clone()`, требует
  проверки с моделью и подходящим драйвером. Для CPU передаётся
  `enable_ynnpack=False` из-за UNIMPLEMENTED на проверенном CPU-пути;
  для GPU параметр YNNPACK в Engine **не передаётся**.
- Один Engine исполняет вызовы последовательно. Начатый
  `Conversation.send_message()` и его native tool-chain не прерываются
  планировщиком посередине.
- Рабочий native launcher включает `shell`, `delegate_task` и четыре
  MQTT-инструмента температуры/влажности. `shell` выполняется с правами
  процесса, не sandbox.
- Web и Voice пока используют старые CORE и их IPC. Переноса истории
  человеческого диалога между CORE нет.
- Старый OpenAI TUI/Web не читает автоматически новый QUERY inbox.
  Native QUERY надо смотреть в `start_tui_tools.sh`.
- Прохождение unit-тестов не доказывает поведение реальной Gemma:
  переключение KV, вызовы инструментов, TASK/QUERY и отказы ещё нужно
  проверять на целевой машине.


## Проверенная конфигурация WSL2 / RTX 3050 (29 октября 2026)

Проверена работа Gemma 4 E2B через LiteRT-LM 0.18.0, GpuArtisan,
Dawn/WebGPU и аппаратную NVIDIA GeForce RTX 3050 Laptop GPU (4 ГБ)
в WSL2. Использована Mesa Dozen 26.0.8, собранная отдельно в
`/opt/mesa-dozen-dawn-test`; системная Mesa не заменялась.

**Ограничение:** в исходном Dozen `fullDrawIndexUint32=false`.
Dawn отказался создавать Vulkan adapter:
`Warning: Vulkan fullDrawIndexUint32 feature required` и
`INTERNAL: No adapters found`. Для эксперимента в
`src/microsoft/vulkan/dzn_device.c` флаг был изменён на `true`,
после чего Dozen пересобран. Это **обход проверки**, а не реализация
возможности и не подтверждение Vulkan conformance; драйвер пригоден
пока только для тестов.

Успешно выполнены следующие проверки:

- Vulkan выбрал `Microsoft Direct3D12 (NVIDIA GeForce RTX 3050 Laptop GPU)`,
  `arch=ampere`, `backend=Vulkan`, `adapterType=Discrete GPU`.
- Минимальный Engine/Conversation: `warm -> OK`.
- Три независимых BASE и три рабочих клона в одном GPU Engine
  (изолированный тест без инструментов):
  MANAGER 400, AGENT_1 183, QUERY MANAGER 153 KV-токенов.
  После работы клонов исходные BASE KV не изменились.
- Native Python Tool Calling на клоне: реальный вызов `gpu_probe`,
  модель вернула контрольный код `RTX3050_TOOL_OK`.
- Native CORE и TUI: обычный диалог, ответ об уличной температуре
  12,53 °C / влажности 59,87%; запрос делегирования AGENT_1 команды
  `pwd` дал ответ `/opt/model`. Последний факт — наблюдение
  ответа модели; для самостоятельного доказательства выполнения
  команды нужен `native-tool RESULT name=shell` в журнале CORE.

Профиль `litert.profiles.e2b-wsl3050` в `cat-agent.yaml`
выбран как активный. Он включает GPU, `vision_backend: none`,
FP16 по умолчанию (без `activation_dtype`), отключённый speculative
decoding и выбор драйвера из
`vulkan_icd_dir: /opt/mesa-dozen-dawn-test/share/vulkan/icd.d`
с `d3d12_adapter: NVIDIA`. Запуск при работающем TASK SYSTEM:

```bash
cd /opt/cat-agent
./start_native_core.sh
```

В другом терминале:

```bash
cd /opt/cat-agent
./start_tui_tools.sh
```

Launcher сам выбирает единственный JSON-файл ICD из каталога
(`VK_DRIVER_FILES`) и экспортирует
`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`. При отсутствующем или
неоднозначном ICD процесс завершается до запуска модели.
Никакие предварительные `export` или длинные CLI-аргументы не нужны.
При переключении на CPU-профиль старые GPU-overrides удаляются из
окружения дочернего процесса. Для CPU выберите `litert.active_profile: e4b`
и перезапустите CORE. Профиль `e2b` остаётся самостоятельным,
без WSL-специфичного драйвера.

Предупреждения `dzn is not a conformant Vulkan implementation` и
`GetProfileSummary UNIMPLEMENTED` наблюдались даже при успешной
генерации. Длительная стабильность, точность вычислений в разных
сценариях и реальная производительность пока не проверены.

## Проверка после изменений

```bash
cd /opt/cat-agent
git pull --ff-only
PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
  -m unittest discover -s tests -v
```

Дополнительные документы: [Task SYSTEM](task-system.md),
[выбор CORE](core-selection.md),
[проверка изоляции KV](core-tools-context-probe.md).
Промежуточные отчёты по этапам удалены из рабочего `docs/`, но остаются
в истории Git.
