# Native CORE — этап 3B: делегирование менеджером через native Tool Calling

## Архитектура

Менеджер получает новый инструмент `delegate_task(agent_id, task)`, если
конфигурация `--agent-config` содержит хотя бы одного агента. Описание
инструмента формируется LiteRT из Python-функции и её docstring.

**Особенность:** функция `delegate_task` **не вызывает модель агента**.
В callback она проверяет ID, текст, бюджет и регистрирует поручение
в очереди текущего хода, возвращая `status=queued`. Это не результат
работы агента.

После завершения менеджером `send_message()`:
1. CORE (в том же единственном потоке модели) извлекает очередь поручений.
2. Запускает разрешённых агентов **последовательно**, каждый на своём
   рабочем `Conversation`.
3. Собирает фактические результаты, без автоматических повторных запусков.
4. Передаёт в диалог менеджера системное сообщение
   `CAT_AGENT_DELEGATION_RESULTS` с результатами JSON.
5. Менеджер генерирует окончательный ответ пользователю.
6. Если в продолжении менеджер запросит ещё одно делегирование, процесс
   повторится, но не больше **трёх** поручений за исходный ход.

Очередь привязана к текущему вызову менеджера и очищается всегда, в том
числе при ошибке генерации. `delegate_task` запрещён во время
прогрева BASE и вне активного хода менеджера. Инструменты агента
разрешаются отдельной allowlist; `delegate_task` агентам недоступен.

После `release` или потери клиента CORE не начинает ещё не исполненные
поручения, если завершившийся инференс успел заметить потерю сеанса.
Уже начатые вызовы LiteRT или Bash при этом **не прерываются**. Это
экспериментальный CORE без полноценной фоновой приоритетной очереди.

## Подготовка к проверке

Использовать **новый** системный промпт менеджера
`prompts_tools/manager.md`. Старый `sys_prompt_manager.md`
предназначен для текстового протокола `/work#` и здесь не годится.
Промпт агента остаётся в `prompts_tools/agent_1.md`.

На WSL остановить уже запущенный тестовый CORE на том же сокете.

```bash
cd /opt/cat-agent
git pull --ff-only

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_orchestration_tools*.py' -v

./start_litert_agent_tools.sh \
  --socket /tmp/cat-agent-litert-tools-test.sock \
  --model /storage/models/litert/gemma-4-E4B-it.litertlm \
  --system-file /opt/cat-agent/prompts_tools/manager.md \
  --agent-config /opt/cat-agent/config/native_agents.example.json \
  --enable-shell \
  --trace-native-json
```

Для теста без MQTT добавить `--disable-mqtt`.
Shell по-прежнему выполняется с правами процесса; при запуске от root
нужно ограничиваться безопасными диагностическими командами.

## Пример IPC-проверки

Во втором терминале:

```bash
python3 - <<'PY'
import json, socket
p = '/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX) as s:
    s.connect(p)
    with s.makefile('r', encoding='utf-8') as rd:
        def query(msg):
            s.sendall((json.dumps(msg, ensure_ascii=False) + '\n').encode())
            reply = json.loads(rd.readline())
            print(reply)
            return reply

        query({'type': 'acquire', 'client': 'delegate-test'})
        for task in (
            'Поручи агенту agent_1 выполнить shell pwd. '
            'Дождись результата агента и сообщи мне каталог.',
            'Поручи агенту agent_1 выполнить shell ls. '
            'Передай мне список файлов из его результата.',
        ):
            accepted = query({'type': 'user', 'text': task})
            if accepted['type'] != 'accepted':
                raise RuntimeError(accepted)
            while True:
                msg = json.loads(rd.readline())
                print(msg)
                if msg['type'] == 'completed':
                    break
        query({'type': 'release'})
PY
```

В логе сервера для каждого задания ожидается:

```text
native-delegation QUEUED id=delegation_1 agent=agent_1
native-delegation START id=delegation_1 agent=agent_1
native-tool CALL name=shell ...
native-delegation FINISH id=delegation_1 agent=agent_1 ok=True
native-delegation REPORT to manager count=1
```

Важнее самого текста модели проверить, что **каждое поручение
выполнено ровно один раз**, агентский вызов следует после первого
`send_message` менеджера и ответ менеджера использует фактический
результат. При ошибке агента менеджер должен сообщить её, а не
утверждать, что команда выполнена.

## Пока не реализовано

Межходовая постоянная очередь, повторные запуски TASK, делегирование
фоновым событиям и кооперативное переключение внутри одного
`send_message()`. Эти свойства относятся к следующему этапу
планировщика. Проверенный в этой версии маршрут используется только
для интерактивного диалога.

Старые CORE, Task SYSTEM и SQLite не изменяются.
