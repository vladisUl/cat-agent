# Native CORE — этап 5B: TASK QUERY → AGENT → отдельный MANAGER

Статус: экспериментальный этап. Старые TASK с `method=query`
поддерживаются через штатный Task SYSTEM RPC, а не через второй планировщик.
Рабочий голосовой интерфейс и пользовательский диалог этим патчем не меняются.

## Путь исполнения

1. `worker.pull` предлагает RUN с `run_id` и `offer_id`.
2. Native CORE читает актуальный сохранённый TASK, проверяет generation,
   method, разрешённые native tools агента.
3. `worker.begin` и `worker.valid` подтверждают право запустить AGENT.
4. AGENT выполняет один автономный `Conversation.ask` и возвращается к BASE.
5. Если `method=query`, отдельная задача приоритета hardware/background
   передаёт результат **tool-free query-manager**. Между двумя вызовами
   Engine может обслужить человека. Перед менеджерским этапом ещё раз
   выполняется `worker.valid`.
6. query-manager использует отдельный прогретый BASE и не имеет ни shell,
   ни MQTT, ни delegate_task. Пользовательский менеджер и его KV не меняются.
7. Краткий результат публикуется как IPC `task_query_result`, сохраняется
   в 32 последних записях оперативной памяти, а TASK завершается через
   `worker.finish`. Ошибки `finish` повторяют только подтверждение.

Для `method=task` прежняя схема 5A сохранена: менеджерский этап не нужен.

## Неподходящие предложения RUN

TASK SYSTEM не владеет разрешениями конкретных AGENT. Для безопасного
планирования добавлен **необязательный** параметр `skip_run_ids` к
`worker.pull`. Старые CORE его не передают и не меняют поведение.

Если RUN не подходит по `skills` или занятости агента:
- Native worker **не вызывает begin или модель**.
- Штатный `finish` над ещё не начатым `offered`-RUN возвращает его в `pending`.
- Только этот worker временно исключает `run_id` из следующего `pull`
  (`skip_run_ids`, срок 2 секунды). Другой исполнитель может его принять.
- При отказе RPC на освобождении предложения worker повторяет только RPC,
  не выполнение. Следующие RUN рассматриваются после подтверждения.
- Не требуется собственная очередь TASK, SQLite или таймер.

TASK SYSTEM по-прежнему единственный владелец очереди RUN.
Протокол совместим с прежними клиентами, но для обхода неподходящих
заданий нужен обновлённый TASK SYSTEM **из этого же коммита**.

## Результаты и границы

Запрос владельца Unix IPC:

```json
{"type":"task_query_results"}
```

Ответ:

```json
{
  "type": "task_query_results",
  "items": [
    {
      "type":"task_query_result",
      "task_id":1,
      "run_id":"...",
      "source":"manual",
      "agent_id":"agent_1",
      "agent_ok":true,
      "text":"..."
    }
  ]
}
```

Если клиент уже подключен как владелец диалога, сервер также отправляет
`task_query_result` **асинхронно**. Клиент обязан различать типы ответов,
а не считать любое прочитанное сообщение ответом на последнюю команду.

Этот журнал **не** сохраняется на диск и **не** является гарантированной
доставкой уведомлений. После рестарта записи пропадают. Это ограничение
нужно устранить перед вводом TASK-уведомлений в эксплуатацию.

## Проверить без модели

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

Тесты проверяют: реальный RPC, два отдельных этапа model inference,
сохранность человеческого KV, отсутствие tools у query-manager,
приоритет человека между шагами, отзыв RUN между шагами, обход
неподходящего RUN, невозможность повтора side effects при сбое `finish`.

## Проверка на Gemma

1. Использовать только изолированную TASK SYSTEM службу из инструкции
   `docs/core-tools-stage5a.md` (с тестовыми SQLite и файлами импорта под
   `/tmp`). Сначала остановить предыдущий тестовый CORE.
2. Запустить native CORE с `--enable-task-worker` и
   `--task-socket /tmp/cat-agent-task-native-test.sock`, настройками
   модели и агента из 5A. Он прогреет ещё один query-manager BASE.
3. В третьем терминале создать `method=query`, затем запустить вручную:

```bash
cd /opt/cat-agent
CAT_AGENT_TASK_SOCKET=/tmp/cat-agent-task-native-test.sock \
PYTHONPATH=/opt/cat-agent/src \
/opt/litert/litert_env/bin/python3 - <<'PY'
from task_system.client import TaskRPC
rpc = TaskRPC()
task = rpc.call(
    'system', 'create_periodic_task',
    'Native QUERY test',
    'Выполни через shell команду pwd и сообщи полный рабочий каталог',
    ['shell'], 3600, method='query', executor='litert',
)
print('TASK ID:', task['task_id'])
print('RUN ID:', rpc.call('system', 'run_task', task['task_id']))
PY
```

В журнале CORE проверить:

```text
native-task ADMITTED ... agent=agent_1 ...
native-tool CALL name=shell ... command='pwd'
native-task EXECUTED ... agent=agent_1 outcome=done
native-task QUERY MANAGER ... kind=reply outcome=done
```

В RPC `system.runs` должен быть один новый RUN с `state=done`.
Результат менеджера можно прочитать через новый IPC-сокет:

```bash
python3 - <<'PY'
import json, socket
p='/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX) as sock:
    sock.settimeout(10)
    sock.connect(p)
    with sock.makefile('r', encoding='utf-8') as rd:
        def send(value):
            sock.sendall((json.dumps(value, ensure_ascii=False)+'\n').encode())
        send({'type':'acquire','client':'query-test'})
        print(json.loads(rd.readline()))
        send({'type':'task_query_results'})
        print(json.dumps(json.loads(rd.readline()), ensure_ascii=False, indent=2))
        send({'type':'release'})
        print(json.loads(rd.readline()))
PY
```

Важно: `task_query_result` появляется только после завершения менеджерского
инференса. Если IPC-клиент был подключён заранее, он мог получить это
сообщение асинхронно ещё до запроса списка.

## Что остаётся

- Надёжная, переживающая рестарт доставка уведомлений.
- Специальное управление длинными TASK QUERY и более тонкая отмена внутри
  одного `send_message` (пока он остаётся непрерываемым).
- Конфликт за один тип исполнителя `litert` между старым и новым CORE:
  на испытаниях использовать только тестовую TASK SYSTEM, не подключать
  рабочие экземпляры без отдельного плана миграции.
