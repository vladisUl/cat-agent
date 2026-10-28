# Native CORE — этап 6A: подключение терминального диалога

Этап 5D проверен на WSL. Этап 6A — **первое подключение пользовательского
интерфейса** к новому native CORE без изменения старого LiteRT/OpenAI CORE,
старых запускателей, голоса и Web.

## Архитектура

- Новый отдельный запуск: `start_tui_tools.sh`.
- Реализация: `agent_core_tools.native_terminal`.
- Переиспользуются существующие `litert_agent.terminal.CoreClient` и
  curses-класс `TerminalTUI`; меняется лишь обработка входящих IPC-ответов
  через отдельный подкласс `NativeTerminalTUI`.
- Сессия `acquire` → `user` → `reply` / `completed` → `release`
  идёт напрямую в native CORE через его Unix socket.
- Команды `snapshot` продолжают показывать статус планировщика.
  Старые поля, отсутствующие в новом CORE, отображаются как неизвестные;
  они **не** заполняются выдуманной статистикой.
- Никакая модель не запускается на стороне терминала; LiteRT Engine,
  KV, инструменты и приоритеты остаются внутри native CORE.

## QUERY-уведомления

Native TUI опрашивает `task_query_results` при подключении и далее
каждые 5 секунд (в том числе когда пользователь ничего не вводит).
Текст каждого нового `run_id` добавляется в историю как `TASK QUERY`.

Ответ менеджера (`reply`) и уведомление (`task_query_result`) могут
приходить в любом порядке. Они не смешиваются и не дублируются.
Повтор `run_id` в live-канале и долговременном списке показывается
**один раз на текущую терминальную сессию**.

После добавления сообщения в терминальную историю клиент отправляет
`task_query_ack` по `run_id`. При потере ACK-ответа повторяется
только подтверждение с интервалом 2 секунды: TASK SYSTEM его
обрабатывает идемпотентно. Получение `task_query_acked=true`
прекращает повторную отправку.

**Важно:** ACK означает, что терминальный клиент принял сообщение в
историю для отображения, **не** подтверждает, что человек лично его
прочитал. Если нужна именно ручная отметка «прочитано», это будет
отдельная политика доставки.

Native TUI не открывает SQLite напрямую и не обращается к TASK SYSTEM
напрямую: все уведомления и ACK проходят через native CORE.

## Запуск на WSL (в отдельном терминале)

Нужно запустить уже испытанные native CORE и TASK SYSTEM, например с
сокетами `/tmp/cat-agent-litert-tools-test.sock` и
`/tmp/cat-agent-task-native-test.sock` (по предыдущим инструкциям 5C).

```bash
cd /opt/cat-agent
git pull --ff-only

./start_tui_tools.sh --socket /tmp/cat-agent-litert-tools-test.sock
```

Если TUI запускается из собственного virtualenv:

```bash
CAT_AGENT_TOOLS_PYTHON=/opt/litert/litert_env/bin/python3 \
 ./start_tui_tools.sh --socket /tmp/cat-agent-litert-tools-test.sock
```

Терминал размером не менее 90×18. Обычная строка отправляется в
native MANAGER. `/quit` закрывает терминальный клиент и освобождает
сессию. Если владелец сокета занят другим интерфейсом, получим `busy`;
принудительный захват чужого диалога запрещён.

При новом подключении TUI запрашивает непрочитанные QUERY-уведомления
из SQLite TASK SYSTEM. Чтобы увидеть этот механизм, создайте тестовый
`method=query` через RPC как в 5C **пока TUI подключен**, затем проверьте
появление `TASK QUERY` и исчезновение RUN из `notifications.list`
после ACK.

## Автоматические проверки без модели

```bash
cd /opt/cat-agent
PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_task_system*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
 -m unittest discover -s tests -p 'test_orchestration_tools*.py' -v
```

Шесть новых тестов `test_agent_core_tools_stage6a.py` проверяют:
диалог MANAGER, одноразовый показ QUERY из live и list, повтор ACK,
неверный формат, отключение сокета и настоящий Unix IPC с fake manager.

## Границы 6A

- Это **только терминальный интерфейс**. Голосовые Vosk/GigaAM/TTS и
  существующая Web UI пока продолжают использовать старый CORE.
- Рестарт терминала не переносит KV человека, если native CORE
  освободил сессию и сбросил Conversation; восстанавливаются только
  непрочитанные QUERY-уведомления из TASK SYSTEM.
- В этом этапе не реализованы streaming-модельные токены, история
  диалога в SQLite, автоматическая миграция между CORE и ручной
  ACK «прочитал».
- Работа 6A на реальной Gemma должна быть подтверждена в WSL
  перед переходом к подключению голоса или Web.
