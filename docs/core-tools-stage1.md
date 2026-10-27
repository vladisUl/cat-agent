# Native tools CORE: этап 1 (IPC-каркас)

Новый CORE строится **параллельно** действующей системе. Входная точка:
`python -m agent_core_tools`. Существующие `agent_core`, `orchestration`,
`litert_agent`, сокеты `litert.sock` и `openai.sock`, Task SYSTEM и SQLite не
изменяются. Новый пакет не импортирует старую реализацию.

## Состояние

**Это только этап 1, не агент с моделью.** Engine, Conversation, менеджер,
агенты, scheduler, Task SYSTEM, MCP, MQTT и shell пока не подключены. В частности,
новый IPC-процесс не регистрирует executor в Task SYSTEM и не забирает RUN.

Каркас реализует часть существующего JSON-lines протокола Unix socket:

| Запрос | Ответ |
| --- | --- |
| `{"type":"ping"}` | `{"type":"pong"}` |
| `{"type":"who"}` | `{"type":"session","owner":null}` либо текущий владелец |
| `{"type":"snapshot"}` | `snapshot` со статусом `NOT_READY` |
| `{"type":"acquire","client":"tui"}` | `acquired` либо `busy` |
| `{"type":"release"}` | `released` |
| `{"type":"user","text":"..."}` | `error`, пока менеджер отсутствует |

Сервис никогда не отправляет фиктивные `accepted`, `reply` или `completed`.
`voice`, `event`, `warm_kv2`, `register_fallback` и остальные операции явно
недоступны на данном этапе.

Сокет по умолчанию: `/run/cat-agent/litert-tools.sock` (режим `0660`).
Путь можно переопределить `--socket`. Эксклюзивная блокировка не даёт
запустить второй CORE на том же сокете; устаревший сокет безопасно удаляется,
действующий или замещённый файл сохраняется. Размер одного входного сообщения
не превышает 64 КиБ; количество одновременных клиентов ограничено 16.

## Запуск

Использовать рабочий Python, где лежит приложение. LiteRT-LM для первого
этапа не требуется.

```bash
cd /opt/cat-agent
git pull --ff-only
CAT_AGENT_TOOLS_PYTHON=/opt/litert/litert_env/bin/python3 \
  ./start_litert_agent_tools.sh
```

Для запуска без прав записи в `/run/cat-agent`:

```bash
CAT_AGENT_TOOLS_PYTHON=/opt/litert/litert_env/bin/python3 \
  ./start_litert_agent_tools.sh --socket /tmp/cat-agent-litert-tools-test.sock
```

В журнале ожидается `TOOLS CORE SKELETON READY: ... (manager not connected)`.
Остановить `Ctrl+C` либо `SIGTERM`. После завершения собственный сокет удаляется.

Проверка из другого терминала:

```bash
python3 - <<'PY'
import json, socket
p = '/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.connect(p)
    with s.makefile('r', encoding='utf-8') as reader:
        for request in ({'type': 'ping'}, {'type': 'snapshot'},
                        {'type': 'acquire', 'client': 'test'},
                        {'type': 'user', 'text': 'Привет'},
                        {'type': 'release'}):
            s.sendall((json.dumps(request, ensure_ascii=False) + '\n').encode())
            print(json.loads(reader.readline()))
PY
```

Тесты каркаса:

```bash
cd /opt/cat-agent
PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_agent_core_tools.py' -v
```

## Следующий этап

Подключить менеджер через `orchestration_tools.contracts.ManagerRuntime`,
использовать `litert_agent_tools` с реальным Conversation, сохранить
стандартные `accepted`/`reply`/`completed` после подключения планировщика.

До программирования полноценного пула контекстов обязательно протестировать:

1. BASE и два независимых клона с разной историей и переключением между ними.
2. Корректное сохранение KV при interleaving на одном Engine.
3. Место приоритетного переключения внутри цепочек нативного Tool Calling.

Не подключать реальный Task SYSTEM и не выдавать READY, пока эти свойства
не подтверждены на установленной модели и LiteRT-LM.
