# Новый native LiteRT CORE и общий TASK SYSTEM

## Промпты вынесены из Git-дерева (миграция)

Единственный параметр в `cat-agent.yaml`:
`runtime.prompt_root: /opt/prompt`. Схема каталогов:

```text
/opt/prompt/
├── prompts/           # OpenAI CORE: sys_prompt_manager.md, mqtt.md, sys_prompt_agent_*.md
└── prompts_tools/     # LiteRT CORE: manager.md, agent_*.md
```

Путь старого CORE экспортируется как `CAT_AGENT_PROMPT_DIR=/opt/prompt/prompts`.
Native LiteRT берёт `manager.md` из `prompts_tools`, а пути AGENT
в `config/native_agents*.example.json` теперь относительные и
разрешаются относительно каталога `manager.md`. Файл
`query_manager.md` больше не используется native CORE. Внешнее
редактирование не меняет Git-файлы, однако промпты применяются лишь при **перезапуске** CORE
(после прогрева BASE).

Перед первым запуском обновлённого кода на WSL скопировать исходные
файлы, не затирая уже существующие:

```bash
cd /opt/cat-agent
install -d /opt/prompt/prompts /opt/prompt/prompts_tools
cp -an prompts/. /opt/prompt/prompts/
cp -an prompts_tools/. /opt/prompt/prompts_tools/
```

Оригиналы в `cat-agent/prompts*` пока сохранены в Git как резерв миграции.
После проверки наличия файлов и успешного запуска обоих CORE можно отдельно
убрать эти копии из репозитория. Отсутствующие внешние промпты **не**
подменяются незаметно файлами из Git: запуск завершается ошибкой.

## Что переделано (октябрь 2026)

Для `cat-agent` создан **второй, независимый LiteRT CORE** с нативным
`Conversation` / Tool Calling вместо текстовых команд `/work#`.
OpenAI CORE остаётся без изменений и продолжает работать через
OpenAI-compatible API. Старый LiteRT CORE сохранён в исходниках,
но штатная служба и launcher переведены на native CORE.

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
./start_system_core.sh
./start_openai_agent.sh
./start_litert_core.sh
```

Подключение TUI:

```bash
./start_tui.sh openai
./start_tui_tools.sh
```

Проверка общего состояния и очереди:

```bash
./start_system_core.sh cores
./start_system_core.sh list
./start_system_core.sh runs
```

`./start_tui.sh litert` использует **старый** LiteRT IPC на `litert.sock`
и не может подключиться к новому native CORE. Для нового CORE используйте
`./start_tui_tools.sh`.

После перехода на native CORE команда запуска одна:
`./start_litert_core.sh`. Промежуточная обёртка
`start_litert_agent_tools.sh` и прежний `start_litert_agent.sh`
удалены. Внутри launcher напрямую исполняется
`/opt/litert-lm-venv/bin/python3 -m agent_core_tools`,
с сохранением `CAT_AGENT_TOOLS_PYTHON` для отладки и тестов.

`systemd/cat-agent-litert.service` также запускает
`start_litert_core.sh`. Его тип — `simple`, а не `notify`,
поскольку native CORE не отправляет `sd_notify(READY=1)`.
При установке systemd-unit вне репозитория его надо обновить
отдельно; `git pull` не перезаписывает `/etc/systemd/system`.
Прежние TUI/Web/Voice для `litert.sock` пока не поддерживают
протокол нового CORE, поэтому для диалога используйте
`start_tui_tools.sh`.

`start_litert_core.sh` берёт активный CPU/GPU-профиль из `cat-agent.yaml`
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
- MANAGER получает четыре нативных MQTT-инструмента, `shell` и
  `delegate_task`. Каждый AGENT **автоматически** получает те же четыре
  read-only MQTT-инструмента, не перечисляя их в JSON. Параметр `tools`
  в `config/native_agents*.example.json` задаёт дополнительные явно
  разрешённые инструменты, в частности `shell`. Он остаётся
  привилегированным: MANAGER имеет Shell при `--enable-shell`, а AGENT
  только при явном разрешении в JSON. `delegate_task` доступен только
  MANAGER. Дополнительной модельной роли QUERY MANAGER больше нет.
  Native TASK worker сверяет навыки с **тем же итоговым набором**
  инструментов AGENT. Shell выполняется с правами процесса, не sandbox.
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
  MANAGER 400, AGENT_1 183, QUERY MANAGER 153 KV-токенов (исторический тест
  до удаления третьей роли; сейчас запускаются только MANAGER и AGENT).
  После работы клонов исходные BASE KV не изменились.
- Native Python Tool Calling на клоне: реальный вызов `gpu_probe`,
  модель вернула контрольный код `RTX3050_TOOL_OK`.
- Native CORE и TUI: обычный диалог, ответ об уличной температуре
  12,53 °C / влажности 59,87%; запрос делегирования AGENT_1 команды
  `pwd` дал ответ `/opt/model`. Последний факт — наблюдение
  ответа модели; для самостоятельного доказательства выполнения
  команды нужен `native-tool RESULT name=shell` в журнале CORE.

Профиль `litert.profiles.e2b-wsl3050` в `cat-agent.yaml`
выбран как активный. **Проверенная текстовая конфигурация** использовала
`backend: gpu` и `vision_backend: none`, FP16 по умолчанию
(без `activation_dtype`), отключённый speculative decoding
и выбор драйвера из
`vulkan_icd_dir: /opt/mesa-dozen-dawn-test/share/vulkan/icd.d`
с `d3d12_adapter: NVIDIA`.

**Эксперимент с Vision GPU (30 октября 2026):** первоначальная
попытка с `gemma-4-E2B-it-gpu.litertlm` и `vision_backend: gpu`
завершилась ошибкой `tf_lite_vision_encoder not found` при создании
Conversation: облегчённая GPU-сборка не предоставила требуемый Vision
encoder. Теперь активному профилю назначен полный файл
`/storage/models/litertlm/gemma-4-E2B-it.litertlm`, при сохранённых
`backend: gpu`, `vision_backend: gpu`, FP16 и настройках Dozen/NVIDIA.
**Работа полной модели с GPU Vision пока не проверена.**
Для возврата к ранее проверенному текстовому режиму нужно восстановить
одновременно файл `gemma-4-E2B-it-gpu.litertlm` и
`vision_backend: none`.

Запуск при работающем SYSTEM CORE:

```bash
cd /opt/cat-agent
./start_litert_core.sh
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

## Разовые и периодические поручения одного AGENT

Native MANAGER использует **один** инструмент:

```python
delegate_task(agent_id, task, period_seconds=0)
```

- `period_seconds=0`: старое разовое поручение с ответом MANAGER.
- `period_seconds=60`: то же самое поручение каждую минуту через
  общий TASK SYSTEM (`executor=litert`, `method=query`).
- `task` сохраняется **без переписывания**. MANAGER исключает из
  него временные условия (они записаны только в `period_seconds`).
  AGENT получает полный текст
  задачи и самостоятельно выбирает из своих разрешённых инструментов
  `get_outdoor_weather`, `shell` и другие. Нет ограничения «один вызов».
- `agent_id` сохраняется в записи TASK. После перезапуска системы задача
  остаётся закреплённой за указанным AGENT; старые задачи без `agent_id`
  по-прежнему используют подбор по `skills`.
- Один тик таймера означает один запуск `NativeAgent.run()`, а не
  Shell-цикл или собственный таймер агента. Запуск может начаться позже,
  если занята единственная модель.

Управление: `list_periodic_queries()`, `stop_periodic_query(task_id)`
(остановить с сохранением записи) и `delete_periodic_query(task_id)`
(удалить запись TASK и таймер, освободив номер) доступны только MANAGER.
Удаление через native MANAGER ограничено периодическими задачами LiteRT;
задачи другого CORE не затрагиваются. Первая активация
происходит через один интервал, не немедленно. AGENT каждый раз
возвращает обязательный статус выполнения (0 — без зарегистрированных
ошибок, не 0 — ошибка) и необязательный текст. Пустой текст не означает
ошибку и не создаёт уведомления. Только непустой результат AGENT
записывается напрямую в SQLite TASK SYSTEM и показывается через native
TUI до ACK. QUERY MANAGER отсутствует, дополнительной генерации нет.
Периодическое задание работает до команды пользователя об остановке.
Неопределённый исход RPC не следует автоматически повторять во избежание
дублированных заданий.

Пример человеческого запроса:

```text
Дай AGENT_1 задание раз в минуту читать температуру на улице
и дописывать измерение в temp.txt.
```

Ожидаемый вызов MODEL: `delegate_task("agent_1", "Полное поручение...",
60)`. Поручение разово отличается только `period_seconds=0`.
Внешние промпты из `/opt/prompt/prompts_tools` не редактируются
программой; краткое описание API инструментов MANAGER добавляется при
инициализации CORE.

Проверка и остановка:

```bash
./start_system_core.sh list
./start_system_core.sh runs
./start_system_core.sh stop <task_id>
tail -f /var/log/litertlm/cat-agent-litert.log
```

## Разовая проверка общих MQTT-инструментов

После обновления Git и перезапуска native CORE в его журнале должны появиться
строки `native-tools MANAGER registered=...` и
`native-tools agent_1 registered=...`; для обоих указывается
`get_outdoor_weather`. В native TUI запрос:
«Поручи AGENT_1 один раз измерить температуру на улице и сообщить
фактическое значение». В LiteRT-журнале ожидается
`MANAGER -> AGENT_1`, затем `native-tool CALL name=get_outdoor_weather`,
`native-tool RESULT ...`, `AGENT_1 -> MANAGER` и ответ пользователю.

Периодическое поручение создаётся тем же `delegate_task`,
но с положительным `period_seconds`.

## Журнал LiteRT CORE

Standalone LiteRT CORE (через `start_litert_core.sh`) теперь пишет
в **собственный** файл `/var/log/litertlm/cat-agent-litert.log`;
файл `/var/log/litertlm/cat-agent.log` принадлежит OpenAI CORE и
не используется LiteRT. Записи также выводятся в stderr/терминал.
Уровень `DEBUG|INFO|WARNING|ERROR|CRITICAL` берётся из
`logging.level` в `cat-agent.yaml` через
`CAT_AGENT_LOG_LEVEL`. Прямой запуск `python -m agent_core_tools`
принимает `--log-level` (уровень из CLI имеет приоритет).
Для изолированного запуска/тестов путь можно переопределить
`CAT_AGENT_LITERT_LOG_PATH` (абсолютный путь).

Журнал содержит технические записи о CORE, прогреве BASE, KV clone,
задачах и инструментах **и** протокол фактических сообщений:
`USER -> MANAGER`, `MANAGER -> USER`, поручения и результаты AGENT,
автономные TASK и результаты AGENT, вызовы Shell с кодом завершения,
размером вывода и временем выполнения, а также данные MQTT. Полный
stdout/stderr Shell намеренно **не дублируется** в журнале: уже
существующий тест закрепляет это ограничение. Ответы MANAGER/AGENT,
сформированные на основании вывода инструментов, журналируются отдельно.
Журнал не раскрывает внутренних промежуточных текстов LiteRT, не
доступных через API. Диагностика нативной библиотеки C++ на stdout/stderr
остаётся в терминале и не перенаправляется в этот Python-файл.

**Важно:** журнал содержит пользовательские сообщения, команды и
результаты инструментов. Файл создаётся с правами `0644` по запросу оператора: все локальные пользователи с доступом к каталогу могут его читать.
При невозможности открыть журнал запуск LiteRT CORE прекращается
с явной ошибкой.

Проверка после запуска:
```bash
tail -f /var/log/litertlm/cat-agent-litert.log
```

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
