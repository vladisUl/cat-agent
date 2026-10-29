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

`./start_tui.sh litert` — **старый** LiteRT CORE на `litert.sock`, а
**не** новый native CORE. Не запускайте старый и новый LiteRT workers
одновременно под одним `executor=litert`.

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


## Проверка GPU-профиля E2B

В `cat-agent.yaml` уже имеется `litert.profiles.e2b` с GPU-моделью
`/storage/models/litertlm/gemma-4-E2B-it-gpu.litertlm`, `backend: gpu`,
`activation_dtype: fp32`, `vision_backend: none`, `speculative: true`.
После помещения файла модели для проверки смените только
`litert.active_profile: e4b` на `e2b`. Запустите `./start_native_core.sh`
обычным способом: launcher передаст GPU, FP32 и speculative в Engine, но **не**
создаст vision executor, так как E2B GPU-модель не содержит секции
`tf_lite_vision_encoder`. Без `vision_backend: none` LiteRT 0.18.0 падает при
`create_conversation()` с `tf_lite_vision_encoder not found`.
Обратный переход — `active_profile: e4b` (CPU).
Никакого автоматического отката на CPU при ошибке GPU нет.
Для первичной диагностики GPU отдельно от speculative можно временно
выставить `speculative: false` в профиле E2B.

**Внимание к фактическому GPU:** запись `Selected adapter: llvmpipe ...\nadapterType=CPU / Software` означает программный Vulkan на CPU, а не
аппаратное GPU-ускорение. Успешный запуск Native CORE с backend=gpu
сам по себе не доказывает работу на физическом GPU. Для ускорения нужен
доступный аппаратный Vulkan/WebGPU-адаптер и его драйвер.

Отладка в случае ошибки:

```bash
cd /opt/cat-agent
git pull --ff-only
./start_native_core.sh
```

После изменения конфигурации или кода перезапустите Native CORE.

Проверки без модели:

```bash
cd /opt/cat-agent
PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \\
  -m unittest discover -s tests -p 'test_native_backend_selection.py' -v
PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \\
  -m unittest discover -s tests -p 'test_shared_native_core.py' -v
```

## Диагностика E2B GPU на llvmpipe (29 октября 2026)

На LiteRT-LM 0.18.0 с программным Vulkan `llvmpipe` зафиксированы
нестабильные `Conversation.send_message("warm")` и `SIGSEGV`.
MANAGER BASE хотя бы в одном запуске прогрелся (`warm -> OK`, 755 KV),
а `ctypes`-clone сохранил 755 токенов. Однако минимальный тест
**без cat-agent, инструментов, clone, speculative decoding и без
нескольких Conversation** также завершился SIGSEGV на первом сообщении.
Это исключает Tool Calling и clone как необходимые условия сбоя.
Ранее добавленные изменения прогрева и промпта AGENT_1 отменены.

В журнале `gdb` зафиксирован SIGSEGV в нативном потоке по адресу
`0x0`, без информативных символов стека. Нельзя уверенно назвать
отказавшую библиотеку на основании этого backtrace. Кроме того,
в журнале явно присутствует `Expected CpuConfig, got GpuArtisanConfig`
при передаче `enable_ynnpack=False` в GPU Engine. Параметр теперь
передаётся **только при backend=cpu** (также исправлены старый
LiteRT CORE и LiteRT server). Нужно повторить минимальный тест и
Native CORE без этого параметра, прежде чем делать вывод о причине
SIGSEGV. Фактический GPU-адаптер в этих опытах — `llvmpipe` (CPU /
Software), не аппаратный GPU.

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
