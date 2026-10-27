# Новый CORE — этап 4A: кооперативный приоритетный планировщик

**Статус: экспериментальный подэтап; не завершение всего этапа 4 ТЗ.**
Существующие `agent_core`, `orchestration`, `litert_agent` и Task SYSTEM
не меняются, автозапуск сохранённых TASK не подключён.

## Исполнение модели

В `agent_core_tools.priority_scheduler.PriorityScheduler` создан
**один** поток работы с LiteRT Engine, заменивший обычный FIFO executor.
Очередь хранит приоритет и порядковый номер для FIFO при равном приоритете:

1. `control` — гарантированный сброс контекстов / cleanup.
2. `human` — интерактивные пользовательские запросы и диагностический agent_task.
3. `hardware` — зарезервирован для будущих событий аппаратуры;
   маршрутизация реальных событий **ещё не подключена**.
4. `background` — отдельные шаги фоновых заданий.

Задачи выбираются по приоритету **только на границах вызовов**. Уже
начатый `Conversation.send_message()` и цепочка автоматических native
tool calls не прерываются и не выполняются параллельно. Это принципиальное
ограничение LiteRT, а не баг планировщика.

Состояние очереди отображается через IPC `snapshot`:
`pending`, `label`, `queued_by_priority` и `background_jobs`.

Контроль допуска ограничивает количество обычных ожидающих операций
(`MAX_PENDING=16`). Очередь не подтверждает неразмещённые запросы.
Служебный сброс KV нельзя отвергнуть из-за заполнения обычной очереди.
При остановке ожидающие операции отбрасываются **без повторного исполнения**.

### Защита сеансов

Для каждой пользовательской операции сохраняется номер эпохи при приёме.
Старый запрос не исполняется после `release` и нового `acquire`, даже
если сокет тот же. Сброс контекста при смене владельца имеет приоритет
`control` и проходит перед очередными интерактивными запросами.

## Фоновые задания — диагностический механизм

Запускается только с `--enable-background-probe` и
`--agent-config`; доступ через тот же защищённый Unix socket
владельцу сеанса. Это **не Task SYSTEM**, не постоянная очередь и не
свободное автономное планирование модели.

Запрос:

```json
{
  "type": "background_task",
  "agent": "agent_1",
  "steps": [
    "Выполни shell pwd и сообщи каталог",
    "Выполни shell ls и перечисли файлы",
    "Повтори текущий каталог через shell pwd"
  ]
}
```

CORE отвечает `background_accepted` с ID и позднее присылает
`background_progress` для каждого выполненного шага и
`background_completed` по завершении. Каждый шаг делает **ровно один**
`chat.ask`, после чего агент сохраняет рабочий KV и переходит в
`WAITING`. Следующий шаг заново ставится в низкоприоритетную очередь.
После последнего шага (или ошибки) рабочий контекст очищается из
отдельного BASE этого агента.

Разрешено 2–8 шагов и не больше одного фонового задания на агента.
Результаты не записываются в SQLite; после остановки процесса пропадают.
`background_task` только диагностический маршрут: планировщик не
выводит последовательность шагов автоматически.

`cancel_background` принимает `background_id` и запрещает новые
шаги. Уже выполняющийся вызов модели / tool side effect при этом
заканчивается, затем контекст агента сбрасывается. При `release`
фоновые задания текущего владельца отменяются аналогично.

## Проверка на WSL

Сначала **остановить** предыдущий тестовый CORE, чтобы не держать две
копии модели.

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
  --enable-shell --enable-background-probe \
  --disable-mqtt
```

Из другого терминала проверить очередь:

```bash
python3 - <<'PY'
import socket, json
p = '/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX) as s:
    s.connect(p)
    with s.makefile('r', encoding='utf-8') as rd:
        def send(msg):
            s.sendall((json.dumps(msg, ensure_ascii=False)+'\n').encode())
        def receive():
            item = json.loads(rd.readline())
            print(item, flush=True)
            return item
        send({'type':'acquire','client':'priority-test'})
        receive()
        send({'type':'background_task', 'agent':'agent_1', 'steps':[
            'Выполни через shell: sleep 2; pwd. Сообщи каталог',
            'Выполни shell ls и перечисли файлы',
            'Выполни shell pwd и сообщи каталог',
        ]})
        accepted = receive()
        assert accepted['type']=='background_accepted'
        send({'type':'user','text':'Скажи одно слово: ГОТОВО. Инструменты не нужны.'})
        completed = False
        background_done = False
        while not (completed and background_done):
            item = receive()
            if item['type']=='completed': completed=True
            if item['type']=='background_completed': background_done=True
        send({'type':'snapshot'})
        receive()
        send({'type':'release'})
        receive()
PY
```

На быстром хосте человек может исполниться до **первого** фонового шага,
поскольку уже ожидающий человек имеет более высокий приоритет.
Для доказательства вытеснения **между** шагами нужно послать пользовательский
запрос во время первого `native-background STEP` (видно в серверном
журнале). Тогда ожидаемый порядок: STEP 1 → USER → STEP 2.

**Проверка корректности:** ответы `background_progress` должны показывать
один и тот же `background_id`, шаги 1/3, 2/3, 3/3, а лог — отдельные
`native-conversation TURN`. После `background_completed` состояние
агента возвращается в `FREE` и KV восстанавливается из BASE.

## Следующая работа

Довести до полного этапа 4: диспетчеризацию аппаратных событий,
управление задачами из менеджера, честную отмену/тайм-ауты, доступность
агентов при долгой работе, а также продуманную обработку длительных
внутренних цепочек Tool Calling. Интеграция автономного Task SYSTEM
выполняется отдельно на этапе 5.
