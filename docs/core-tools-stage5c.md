# Native CORE — этап 5C: SQLite QUERY inbox и явное ACK

Это надстройка над проверенными 5A/5B. Единственный владелец данных
— существующий TASK SYSTEM. Native CORE не открывает свою БД.

## Точная семантика доставки

1. Native AGENT выполняет QUERY один раз и возвращает результат менеджеру.
2. Отдельный tool-free QUERY MANAGER формирует текст и возвращается к BASE.
3. Native worker держит результат в оперативном outbox и повторяет
   **только** штатный RPC `worker.publish_result`, пока TASK SYSTEM не
   подтвердит сохранение в своей SQLite.
4. TASK SYSTEM сохраняет единственную неизменяемую запись по `run_id`
   (также привязанную к `owner`, `peer`, `offer_id`) **в той же транзакции**,
   что и другие поля состояния. Повтор того же RPC идемпотентен; подмена
   уже записанного текста отклоняется.
5. Только после успешной записи Core может отправить live IPC уведомление
   `task_query_result` и вызвать `worker.finish`. Если `finish` потеряет
   ответ, повторяется только `finish`, без нового инструмента и модели.
6. Любой авторизованный IPC-клиент владельца может прочитать непрочитанные
   уведомления из TASK SYSTEM через `task_query_results` и подтвердить
   конкретное через `task_query_ack`. Одна только выдача списка или live
   отправка **не** отмечает уведомление прочитанным.

**Граница гарантии**: после успешного коммита SQLite уведомление остаётся
доступным после перезапуска обеих служб, пока клиент не сделает ACK.
Доставка имеет семантику *at least once*: до ACK сообщение может читаться
неоднократно. Чтобы не показывать один RUN дважды, клиент должен
дедуплицировать по `run_id`.

Если CORE погиб **после ответа модели, но до первого успешного
`publish_result`**, этот ответ ещё не сохранён, и восстановить его
невозможно. TASK SYSTEM помечает уже начатый RUN как `interrupted` и не
запускает его заново автоматически. Это сознательный предел 5C, а не
обещание абсолютной доставки при любом моменте падения.

## RPC TASK SYSTEM

Добавлена совместимая операция:

```text
worker.publish_result(owner=..., run_id=..., offer_id=..., result={
    "type":"task_query_result", "run_id":"...",
    "task_id":1, "source":"manual", "agent_id":"agent_1",
    "agent_ok":true, "text":"..."
})
```

Действующий RPC доступен только исполнителю, владеющему RUN.
TASK SYSTEM проверяет состояние `running`, актуальность TASK и
`method=query`. Уже сохранённая запись может быть подтверждена повторным
вызовом с теми же owner/peer/offer/result даже после `finish`.

Для чтения и ACK (через существующий защищённый Unix RPC):

- `notifications.list(limit=100)` — непрочитанные сообщения, максимум
  100 за вызов. Чтение не изменяет состояние.
- `notifications.ack(run_id)` — помечает уведомление полученным,
  возвращает true. Повторный ACK сохраняет true, пока запись есть.
  Неизвестный RUN возвращает false.

Хранилище ограничено **1024 непрочитанными** сообщениями. При заполнении
новые записи не удаляют старые: RPC сообщает ошибку, рабочий outbox
делает повторную попытку. Подтверждённых записей для истории остаётся
до 128; самые старые подтверждённые записи допускается удалять.
БД сохраняет записи в существующей SQLite (в поле JSON-состояния TASK
SYSTEM), без новой схемы SQLite и без второй БД.

## IPC native CORE

```json
{"type":"task_query_results"}
```

Ответ `task_query_results` содержит массив `items` из долговременного
хранилища TASK SYSTEM, а не диагностический cache CORE.
Если Task SYSTEM недоступен, возвращается ошибка (не пустой список).

```json
{"type":"task_query_ack","run_id":"RUN_ID_ИЗ_УВЕДОМЛЕНИЯ"}
```

Ответ:

```json
{"type":"task_query_acked","run_id":"RUN_ID_ИЗ_УВЕДОМЛЕНИЯ","acknowledged":true}
```

Обе команды требуют уже приобретённую human IPC-сессию `acquire`;
прочие клиенты получают `busy`. Доступ к самому Unix RPC TASK SYSTEM
по-прежнему ограничивается разрешениями на сокет. В этом этапе не
вводится привязка уведомления к отдельному конечному пользователю:
владелец общей IPC-сессии может подтвердить результат.

## Проверки на WSL

```bash
cd /opt/cat-agent
git pull --ff-only

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_orchestration_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_task_system*.py' -v
```

В частности, `test_task_system_inbox.py` проверяет:
- перезапуск SQLite после `finish` и **между publish и finish**;
- idempotent publish после завершения RUN;
- запрет публикации чужим исполнителем или с неверным offer ID;
- отмену TASK, запрет публикации у `method=task`;
- откат при исключении сохранения SQLite;
- ограничения структуры сообщения, явный ACK.

`test_agent_core_tools_task_worker.py` теперь также проверяет потерянный
ответ до и после коммита `publish_result` и отсутствие повторного AGENT.
`test_agent_core_tools_stage5c.py` проверяет IPC через Unix socket и
разрешение команд только владельцу.

## Проверка на реальной Gemma

Пользуемся **отдельными тестовыми** TASK SYSTEM/SQLite/сокетом из
`docs/core-tools-stage5a.md` и настройками native CORE из
`docs/core-tools-stage5b.md`. Рабочие базы и TASK не использовать.

1. Остановить старый тестовый CORE и выполнить `git pull --ff-only`.
2. Перезапустить обе тестовые службы из обновлённого кода.
3. Создать и запустить один `method=query` через `run_task`
   (готовая команда есть в `docs/core-tools-stage5b.md`).
4. На CORE убедиться в последовательности:

```text
native-task EXECUTED ... outcome=done
native-task QUERY MANAGER ... kind=reply outcome=done
native-task QUERY STORED run=...
```

5. Сначала прочитать из TASK SYSTEM напрямую (если CORE ещё работает):

```bash
CAT_AGENT_TASK_SOCKET=/tmp/cat-agent-task-native-test.sock \
PYTHONPATH=/opt/cat-agent/src \
/opt/litert/litert_env/bin/python3 - <<'PY'
from task_system.client import TaskRPC
rpc = TaskRPC()
print('UNREAD:', rpc.call('notifications', 'list'))
print('RUNS:', rpc.call('system', 'runs'))
PY
```

6. Остановить и **снова поднять только native CORE**; повторить
   `notifications.list` — уведомление должно остаться.
7. Сделать IPC `acquire` → `task_query_results` → после обработки сообщения
   `task_query_ack(run_id)` → повторить `task_query_results`.
   После ACK этот RUN больше не должен появляться среди непрочитанных.

**Не удалять и не переинициализировать тестовую SQLite между пунктами 5–7.**
Для проверки рестарта самого Task SYSTEM можно остановить и запустить
тестовый сервер с тем же путём SQLite.

## Следующие задачи

- Надёжная привязка и доставка к конкретному интерфейсу (голос/чат)
  с собственными правилами чтения и дедупликации.
- Восстановление ещё не подтверждённого *модельного* результата после
  аварии самого CORE до записи в Task SYSTEM потребует отдельной
  WAL/outbox-архитектуры либо более узкого гарантийного контракта.
- Диспетчеризация старого и нового CORE в общем производственном TASK
  SYSTEM требует отдельного этапа миграции и испытаний.
