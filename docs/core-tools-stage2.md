# Native-tools CORE — этап 2: менеджер с LiteRT Conversation

Этап 1 сохранён: запуск без `--model` по-прежнему создаёт пустой
диагностический CORE с `NOT_READY`. Существующие каталоги и Task SYSTEM
не изменены.

## Реализовано

Добавлены `orchestration_tools.manager.NativeManager` и
`orchestration_tools.litert_runtime.build_native_manager`. Последний
использует проверенный `litert_agent_tools.NativeConversationChat`:
один Engine, прогретый BASE, один рабочий клон, автоматический нативный Tool
Calling, четыре MQTT-функции и опциональный `shell(command)`.

`agent_core_tools.server.CoreServer` теперь умеет принимать работающий
менеджер и обрабатывает пользовательские запросы в **одном** рабочем потоке
модели. IPC остаётся отзывчивым при генерации: `user` получает
`accepted`, а по завершении `reply` и `completed` с тем же
`request_id` и `session_id`.

При `release` или отключении клиента очистка рабочего диалога
ставится в очередь **после** уже начавшегося обращения к модели.
Ответ старого владельца не доставляется новому. Ошибка генерации
вызывает восстановление из BASE, без автоматического повтора запроса.
Пустой ответ помечается как `silent`, с `completed` без `reply`.

**Ограничения:** пока работает только один владелец (TUI-подобный
`acquire/user/release`). Нет пула агентов, очереди приоритетов, Voice,
фоновых задач, регистрации исполнителя в Task SYSTEM и интеграции MCP.
`READY` в этом CORE означает лишь наличие прогретого менеджера,
**не** готовность исполнителя TASK: сервис Task SYSTEM о нём ничего не знает.
Запрос в процессе выполнения не прерывается при отключении клиента.

## Проверка на WSL

Перед проверкой остановить ранее запущенный демон этапа 1, если он
слушает тот же сокет. Старый LiteRT CORE останавливать не требуется.

```bash
cd /opt/cat-agent
git pull --ff-only

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v
```

Запуск (другой терминал):

```bash
cd /opt/cat-agent
./start_litert_agent_tools.sh \
  --socket /tmp/cat-agent-litert-tools-test.sock \
  --model /storage/models/litert/gemma-4-E4B-it.litertlm \
  --system-file /opt/prompt/sys_prompt_manager.md \
  --enable-shell \
  --trace-native-json
```

`--enable-shell` даёт модели все права пользователя, под которым
запущен процесс; shell по умолчанию стартует в `/opt/model`.
Это экспериментальный локальный процесс, не открывать сокет
посторонним пользователям. `--disable-mqtt` исключает MQTT-инструменты
из описания BASE. При ошибке создания модели IPC не запускается.

Из второго терминала:

```bash
python3 - <<'PY'
import socket, json
path = '/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX) as s:
    s.connect(path)
    with s.makefile('r', encoding='utf-8') as rd:
        def ask(msg):
            s.sendall((json.dumps(msg, ensure_ascii=False) + '\n').encode('utf-8'))
            answer = json.loads(rd.readline())
            print(answer)
            return answer
        ask({'type':'ping'})
        ask({'type':'acquire', 'client':'test'})
        for prompt in ('Привет, кто ты?', 'Выполни shell pwd и скажи результат'):
            accepted = ask({'type':'user', 'text':prompt})
            if accepted['type'] != 'accepted':
                raise RuntimeError(accepted)
            while True:
                result = json.loads(rd.readline())
                print(result)
                if result['type'] == 'completed':
                    break
        ask({'type':'release'})
PY
```

В логе CORE должны появиться `native-conversation BASE-WARM`,
`BASE-CLONE`, `native-tool CALL name=shell` и информация о ходе
Conversation. `reply` должен вернуть `/opt/model` для `pwd`.

## Прежде чем вводить пул агентов

Нужны отдельные эксперименты на реальном LiteRT 0.18.x с *двумя*
долгоживущими клонами: проверить независимость фактического содержимого
KV после переключения A → B → A. Это не доказано ни первым клоном, ни
счётчиком токенов. Затем определить безопасные точки приоритетного
переключения в цикле нативных инструментов.
