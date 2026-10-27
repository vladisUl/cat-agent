# Native CORE — этап 5A: штатный TASK SYSTEM RPC (начальная интеграция)

## Границы реализации

Новый CORE подключается к **существующему** Task SYSTEM как отдельный
исполнитель категории `litert`. Собственного SQLite, таймеров и MQTT
мониторов он **не создаёт**. TASK SYSTEM по-прежнему единственный владелец
очереди RUN, подтверждений и событий, а новый CORE владеет **только**
последовательным выполнением Native Conversation на одном Engine.

**Этап 5A поддерживает сохранённые TASK с `method="task"`.**
`method="query"` в этой итерации не запускается: в старом CORE он
имеет дополнительную семантику получения результата и доставки
менеджеру. Нельзя без проверки считать его обычным TASK.
В частности, команда `python -m task_system.cli create` в действующем
коде создаёт **`method=query`**. Для этого испытания используем
прямой штатный вызов RPC `create_periodic_task(..., method="task")`.

## Протокол

```text
TASK SYSTEM (свой SQLite / таймеры / MQTT)
  worker.heartbeat(owner, executor=litert, ready=true)
  worker.pull(owner, executor=litert, seen=[...])
  tasks.get(task_id) -> актуальные skills, text, generation
  |
  | Приём только если агент свободен и его tools включают все TASK skills
  v
CORE priority scheduler (один native model worker)
  worker.begin(run_id, offer_id) -> разрешение
  worker.valid(run_id, offer_id) -> свежая проверка
  AGENT.run(task.text) -> Нативный Tool Calling
  |
  | completion OUTBOX в памяти
  v
TASK SYSTEM: worker.finish(run_id, offer_id, outcome)
```

- Вызовы модели строго последовательные, через существующий
  `PriorityScheduler`: `source=mqtt` получает `hardware`,
  прочие сохранённые RUN — `background`.
- При недоступном RPC до `begin` модель **не вызывается**.
  Проверка `valid` обязательна непосредственно перед инференсом.
- Ошибка ответа RPC `finish` повторяет **только подтверждение**, не
  запуск модели или инструмента. `finish` идемпотентен по offer ID.
- Отдельный поток RPC обновляет heartbeat раз в 5 секунд, в том числе
  пока модель выполняет длительный `send_message()`.
- Права берутся из списка разрешённых нативных инструментов агента,
  а содержимое задания — из сохранённого `tasks.get`, с проверкой
  актуального `generation`. Недоступные skills не исполняются.
- Контекст каждого успешно завершившего одноходовый TASK агента
  сбрасывается из его собственного BASE.
- Пул допускает один зарезервированный TASK на агента. Диагностические
  `background_task` / `hardware_probe` не могут занять уже
  зарезервированного TASK исполнителя.
- При потере процесса RUN, который уже начался, не должен запускаться
  автоматически второй раз: это обеспечивает существующий TASK SYSTEM.

Существует важная граница: один `send_message()` не прерывается
посередине. `max_runtime_seconds` диагностических фоновых заданий
не распространяется на RUN TASK SYSTEM; тайм-аут исполнительного
модельного вызова ещё не внедрён.

## Безопасная отдельная проверка на WSL

Чтобы **не трогать рабочие TASK и SQLite**, поднимем отдельный
тестовый TASK SYSTEM с сокетом и БД под `/tmp`.
Остановить старый тестовый native CORE на
`/tmp/cat-agent-litert-tools-test.sock` перед началом.

### Терминал 1 — тестовый TASK SYSTEM

```bash
cd /opt/cat-agent
git pull --ff-only

PYTHONPATH=/opt/cat-agent/src \
/opt/litert/litert_env/bin/python3 - <<'PY'
from pathlib import Path
import threading
import time
from task_system.engine import TaskEngine
from task_system.main import Server, Handler

root = Path('/tmp')
db = root / 'cat-agent-task-native-test.sqlite3'
socket_path = root / 'cat-agent-task-native-test.sock'
# IMPORTANT: explicit /tmp paths prevent importing production legacy files.
engine = TaskEngine(
    db, task_file=root / 'cat-agent-task-native-legacy.txt',
    event_file=root / 'cat-agent-task-native-events.json',
)
try:
    with Server(str(socket_path), Handler) as srv:
        srv.engine = engine
        worker = threading.Thread(target=srv.serve_forever, daemon=True)
        worker.start()
        print('TASK_SYSTEM_READY:', socket_path, flush=True)
        try:
            while True:
                engine.tick()
                time.sleep(.25)
        except KeyboardInterrupt:
            pass
        finally:
            srv.shutdown()
            worker.join()
finally:
    engine.close()
PY
```

Дождаться `TASK_SYSTEM_READY`. Не использовать производственную
БД или производственный сокет на этом испытании.

### Терминал 2 — native CORE

```bash
cd /opt/cat-agent
./start_litert_agent_tools.sh \
  --socket /tmp/cat-agent-litert-tools-test.sock \
  --model /storage/models/litert/gemma-4-E4B-it.litertlm \
  --system-file /opt/cat-agent/prompts_tools/manager.md \
  --agent-config /opt/cat-agent/config/native_agents.example.json \
  --enable-shell --disable-mqtt \
  --enable-task-worker \
  --task-socket /tmp/cat-agent-task-native-test.sock \
  --trace-native-json
```

Ожидается `native-task WORKER READY`. Если TASK SYSTEM недоступен,
CORE откажется включать worker, а не будет имитировать READY.

### Терминал 3 — создать TASK и вручную запустить RUN

```bash
cd /opt/cat-agent
CAT_AGENT_TASK_SOCKET=/tmp/cat-agent-task-native-test.sock \
PYTHONPATH=/opt/cat-agent/src \
/opt/litert/litert_env/bin/python3 - <<'PY'
from task_system.client import TaskRPC
rpc = TaskRPC()
task = rpc.call(
    'system', 'create_periodic_task',
    'Проверка native TASK', 'Выполни через shell команду pwd и сообщи каталог',
    ['shell'], 3600,
    method='task', executor='litert',
)
print('CREATED TASK:', task['task_id'], 'method:', task['method'])
print('RUN:', rpc.call('system', 'run_task', task['task_id']))
PY
```

В терминале CORE ожидаются:

```text
native-task ADMITTED run=... offer=... agent=agent_1 priority=background
native-tool CALL name=shell cwd=/opt/model command='pwd'
native-task EXECUTED run=... agent=agent_1 outcome=done
```

В журнале TASK SYSTEM — `TASK offered`, `TASK running`,
`TASK finished ... state=done`.

**Проверка статуса RPC**:

```bash
CAT_AGENT_TASK_SOCKET=/tmp/cat-agent-task-native-test.sock \
PYTHONPATH=/opt/cat-agent/src \
/opt/litert/litert_env/bin/python3 - <<'PY'
from task_system.client import TaskRPC
rpc = TaskRPC()
print('CORES:', rpc.call('system', 'cores'))
print('RUNS:', rpc.call('system', 'runs'))
PY
```

Ожидается ровно один RUN `done` с данным ID, без повторного запуска
`pwd`. Затем остановить оба тестовых процесса.

## Тесты без LiteRT

```bash
cd /opt/cat-agent
PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_orchestration_tools*.py' -v
```

В новом `test_agent_core_tools_task_worker.py` проверяется выдача
реального RUN из `TaskEngine`, `begin/valid/finish`, запрет
повтора действий при ошибке `finish`, остановка TASK до `begin`,
ограничение инструментов, приоритет человека, отказ при недоступном
`begin`, а также настоящий Unix socket RPC.

## Оставшиеся ограничения этапа 5

- Сохранённые TASK `method=query` пока не исполняются native-worker.
  Их нельзя тестировать описанным здесь скриптом без изменения метода.
- Результаты TASK пока только подтверждаются через RPC; механизм
  автономного уведомления пользователя не подключён.
- Отказ процесса после выполнения side effect, но до доставки `finish`
  остаётся `interrupted`, а не автоматически повторяется — это важная
  защита.
- Нет восстановления результата outbox после *перезапуска самого CORE*;
  защитой от повторов остаётся состояние RUN в TASK SYSTEM.
- Для полной миграции старого CORE необходимо отдельно проверить
  конкурентную работу исполнителей и обработку `query`/уведомлений.
