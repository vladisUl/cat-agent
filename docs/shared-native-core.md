# Совместный TASK SYSTEM: старый OpenAI CORE + native LiteRT CORE (этап 7A)

## Схема

Один экземпляр **штатного** `task_system.main`:

```text
                         /run/cat-agent/tasks.sock
                   TASK SYSTEM / одна SQLite / MQTT / RUN
                         /                 \
             worker=openai                 worker=litert
                    /                         \
   OpenAI CORE (без изменений)          native LiteRT CORE
   /run/cat-agent/openai.sock            /run/cat-agent/litert-tools.sock
          |                                       |
    старый Web/TUI                           start_tui_tools.sh
```

- Сервис запускается **один раз**: `./start_task_system.sh`.
- Старый OpenAI CORE запускается **без изменений**: `./start_openai_agent.sh`.
- Новый native CORE запускается `./start_native_core.sh` и подключается
  к *этому же* `/run/cat-agent/tasks.sock`. У него **нет** своей SQLite,
  таймеров, MQTT-monitor и списка TASK.
- Оба CORE работают независимо; native CORE имеет свой один LiteRT Engine
  и собственный сериализованный scheduler.
- База существующего TASK SYSTEM остаётся
  `/var/lib/cat-agent/task-system.sqlite3`.
  Этап не изменяет её схему и **не** импортирует тестовую БД из `/tmp`.
- Старые `start_litert_agent.sh` и `start_litert_agent_tools.sh`
  не изменены. Первый запускает **старый** LiteRT CORE,
  второй остаётся низкоуровневым запускателем нового CORE с явными
  `--model`, `--socket`, `--task-socket` для экспериментов.

## Запуск (разные терминалы)

Перед подключением к рабочему TASK SYSTEM **убедиться**, что его
реальная SQLite и сохранённые TASK соответствуют ожиданиям:
при READY-исполнителях существующие периодические и MQTT-задания
могут автоматически начать выполняться. Не запускать параллельно
второй TASK SYSTEM на той же базе или том же сокете.

Остановить тестовый CORE и тестовый TASK SYSTEM на сокетах из `/tmp`.
Их база остаётся нетронутой; удалять её необязательно.

Терминал 1 (только если штатный сервис ещё не запущен):

```bash
cd /opt/cat-agent
git pull --ff-only
./start_task_system.sh
```

Ожидаем `TASK_SYSTEM_READY: /run/cat-agent/tasks.sock`.
Не запускать второй экземпляр сервиса.

Терминал 2:

```bash
cd /opt/cat-agent
./start_openai_agent.sh
```

Native CORE выполняет проверку `system.ping` общего TASK SYSTEM **до**
загрузки модели. Если штатный сервис отсутствует, запуск прерывается
без прогрева LiteRT.

Терминал 3:

```bash
cd /opt/cat-agent
./start_native_core.sh
```

Эта команда читает выбранный `litert.active_profile` из
`cat-agent.yaml` через **существующий** `orchestration.config_env`.
Native runtime поддерживает здесь **CPU**; GPU-профиль отклоняется
до запуска модели. Настройки speculative и YNNPACK старого runtime
не переносятся в native и не заявляются как поддержанные.

Параметры по умолчанию:

| Параметр | Значение |
|---|---|
| Native socket | `/run/cat-agent/litert-tools.sock` |
| TASK SYSTEM socket | `/run/cat-agent/tasks.sock` |
| Model | `litert.profiles.<active>.model` |
| CPU threads | `litert.profiles.<active>.cpu_threads`, если задано |
| MANAGER prompt | `prompts_tools/manager.md` |
| Agents | `config/native_agents.example.json` |
| Shell CWD | `runtime.workspace` из `cat-agent.yaml` |
| Tools | `shell`, native `delegate_task`; MQTT tool пока отключён |
| Worker | `--enable-task-worker`, executor=`litert` |

Если модель на WSL физически расположена не по пути,
указанному в `cat-agent.yaml`, **не нужно** ради этого
менять путь старому CORE; используйте:

```bash
CAT_AGENT_NATIVE_MODEL_PATH=/storage/models/litert/gemma-4-E4B-it.litertlm \
 ./start_native_core.sh
```

Значение выше приведено из тестовых инструкций 5A/5C: фактическое
расположение файла должно совпадать с данным компьютером.
Изменяемые параметры среды:
`CAT_AGENT_NATIVE_MODEL_PATH`, `CAT_AGENT_NATIVE_CORE_SOCKET`,
`CAT_AGENT_NATIVE_MANAGER_PROMPT`, `CAT_AGENT_NATIVE_AGENT_CONFIG`,
`CAT_AGENT_NATIVE_SHELL_CWD`, `CAT_AGENT_TASK_SOCKET`,
`CAT_AGENT_TOOLS_PYTHON`. При ошибке отсутствующего файла
native CORE не запускается. Конфликт его сокета с
`/run/cat-agent/openai.sock`, `/run/cat-agent/litert.sock`
или TASK SYSTEM отклоняется.

Терминал 4 — новый чат, отдельно от OpenAI TUI:

```bash
cd /opt/cat-agent
./start_tui_tools.sh
```

Старый `./start_tui.sh` и selector `scripts/select_core.sh`
сейчас **не выбирают native CORE**: имя `litert` там означает
прежний LiteRT CORE и сокет `/run/cat-agent/litert.sock`.
Не перенаправлять туда Web/Voice, пока их IPC-функции не
адаптированы. Старый OpenAI интерфейс можно запустить отдельно:
`./start_tui.sh openai` (но только один TUI-владелец на CORE).

## Распределение TASK

Проверить регистрации:

```bash
cd /opt/cat-agent
./task_system.sh cores
```

При доступности обеих моделей:

```text
openai READY
litert READY
```

Политика существующего TASK SYSTEM не меняется:

- `executor=auto`: выбирает READY OpenAI, иначе READY LiteRT.
  При отсутствии READY оба оставляют RUN в `pending`.
- `executor=litert`: исполняет native CORE, **если только**
  старый LiteRT CORE не запущен параллельно. Оба зарегистрировались
  бы под одним backend `litert`. Пока для миграции запускайте
  только **один** из двух LiteRT CORE.
- `executor=openai`: только старый OpenAI CORE.
- `offered` может переиздаваться после исчезновения executor;
  начатый `running` не переигрывается после падения CORE.
- Старые TASK с фиксированным `executor=litert` не переписываются
  автоматически. Native AGENT принимает лишь skills, разрешённые
  в его JSON-конфигурации; неподдерживаемые задания отложит.

Проверка без опасных side effects: сначала посмотреть
`./task_system.sh list`, затем на **разовом безопасном**
`shell pwd` создать отдельный TASK (с удалением/остановкой
по окончании). Не менять исполнителей существующих рабочих задач
ради проверки.

## Уведомления и границы

Native QUERY-отчёты сохраняются в **общей** SQLite TASK SYSTEM.
Новый `start_tui_tools.sh` получает их через native CORE и
подтверждает `task_query_ack` по `run_id`.

**Пока старый OpenAI TUI/Web не получает автоматически**
`notifications.list` из этого нового общего QUERY inbox. Поэтому
native-уведомления следует читать через native TUI или через RPC.
Это ограничение интерфейса, а не разделение SQLite.

Звук, Web и автоматическое переключение пользовательской сессии
между двумя CORE в 7A не реализуются. Связь TASK SYSTEM не
передаёт контекст личного диалога между моделями.

## Автоматические тесты

```bash
cd /opt/cat-agent
git pull --ff-only
PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_shared_native_core.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_task_system*.py' -v
```

`test_shared_native_core.py`: совместное состояние READY,
приоритет `auto`, закрепление `litert`, fallback при
NOT_READY OpenAI и общий SQLite QUERY inbox; также проверка
разбора параметров запускателя, CPU-only, раннего TASK SYSTEM ping
и запрета конфликтующих сокетов. Все тесты используют фальшивую модель и временную
SQLite, **никакого** рабочего сервиса не запускают.

Работоспособность обоих реальных CORE на одной машине и
доставка живых QUERY проверяются отдельно на WSL после тестов.
