# Новый CORE — этап 4B: вход аппаратных событий и жизненный цикл долгих заданий

**Экспериментальный подэтап.** Это не подключение продукционных событий
Task SYSTEM: ни SQLite, ни подписки MQTT, ни регистрация исполнителя RUN
здесь не меняются. Старый CORE остаётся независимым.

## 1. Приоритетное аппаратное событие

Добавлен диагностический IPC-запрос `hardware_probe`. Он доступен
только владельцу сеанса и только при флаге `--enable-hardware-probe`.
Поля `source`, `name`, `agent` и `text` строго проверяются.

```json
{
  "type": "hardware_probe",
  "source": "gpio",
  "name": "door",
  "agent": "agent_2",
  "text": "Диагностическое событие gpio/door. Сообщи, что получил его."
}
```

Сервер отвечает `hardware_accepted` с отдельным `event_id`.
Само выполнение происходит через очередь `hardware`, **после**
ожидающих `human`, но **до** следующих фоновых шагов. По результату
отправляется `hardware_completed` с `ok` и `text`.

Событие не выдаётся за настоящий GPIO/MQTT сигнал и не вызывает
`begin_run`/`finish_run` Task SYSTEM. Если агент уже занят
фоновым заданием, вход отклоняется до постановки в очередь. Вложенного
инференса и параллельных операций на одном Engine нет.

Для проверки разных агентов добавлены `agent_2.md` и
`native_agents_2.example.json`. У второго агента нет shell:
он отвечает только текстом на тестовое событие.

## 2. Состояние фона и ограничения

Запрос `background_task` дополнительно принимает
`max_runtime_seconds` от 1 до 3600 (по умолчанию 120).

Он **не прерывает** выполняющийся `Conversation.send_message()`.
При истечении времени до очередного обращения новый шаг не начинается;
сохранённый KV возвращается к BASE. Уже выполненные инструментальные
операции **никогда не повторяются автоматически**.

Добавлены состояния:
`QUEUED` → `RUNNING` → `WAITING` → `COMPLETED`,
а также `CANCELLING`, `CANCELLED`, `TIMED_OUT`, `FAILED`.

В запросе `background_status` с `background_id` можно получить
текущее или терминальное состояние. Хранятся **32 последних**
завершённых заданий в оперативной памяти, видимые только владельцу
соответствующей эпохи сеанса. При рестарте CORE этот реестр исчезает;
это не аналог TASK SYSTEM.

При отмене и истечении срока отправляется
`background_completed` с `ok=false` и полем `state`,
если пользовательский сеанс ещё доступен.

## 3. Запуск на WSL

Остановить предыдущий тестовый CORE на том же сокете.

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
  --agent-config /opt/cat-agent/config/native_agents_2.example.json \
  --enable-shell --enable-background-probe --enable-hardware-probe \
  --disable-mqtt
```

`--enable-shell` выполняет команды с полномочиями текущего ОС-пользователя.
Для этой проверки допустимы только простые диагностические команды.

## 4. Один тестовый клиент

В другом терминале:

```bash
python3 - <<'PY'
import socket, json, time

p = '/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
    s.settimeout(90)
    s.connect(p)
    with s.makefile('r', encoding='utf-8') as rd:
        def send(item):
            s.sendall((json.dumps(item, ensure_ascii=False)+'\n').encode())
        def recv():
            obj = json.loads(rd.readline())
            print(obj, flush=True)
            return obj

        send({'type':'acquire','client':'stage4b-test'})
        recv()
        send({'type':'background_task','agent':'agent_1',
              'max_runtime_seconds':120,
              'steps':[
                  'Выполни через shell команду sleep 3; pwd. Сообщи каталог.',
                  'Выполни через shell ls. Перечисли файлы.',
                  'Выполни через shell pwd. Сообщи каталог.'
              ]})
        accepted = recv()
        assert accepted['type']=='background_accepted', accepted
        job_id = accepted['background_id']

        # Убедимся, что первый фоновой шаг уже начал работать.
        for _ in range(150):
            send({'type':'snapshot'})
            msg = recv()
            if msg['type']=='snapshot' and msg['status']['label']=='background':
                break
            time.sleep(0.02)
        else:
            raise RuntimeError('Не удалось зафиксировать активный фоновый шаг')

        send({'type':'hardware_probe','source':'gpio','name':'door',
              'agent':'agent_2',
              'text':'Диагностическое событие gpio/door. Подтверди получение.'})
        hw = recv()
        assert hw['type']=='hardware_accepted', hw

        send({'type':'user','text':'Ответь одним словом ГОТОВО, без инструментов.'})
        human = recv()
        assert human['type']=='accepted', human

        completions = set()
        while len(completions) < 3:
            m = recv()
            if m['type']=='completed':
                completions.add('human')
            elif m['type']=='hardware_completed':
                completions.add('hardware')
            elif m['type']=='background_completed':
                completions.add('background')

        send({'type':'background_status','background_id':job_id})
        status = recv()
        assert status['type']=='background_status', status
        assert status['job']['state']=='COMPLETED', status
        send({'type':'release'})
        recv()
PY
```

**Ожидаемый приоритет** при соблюдении условия, что первый фоновый
шаг уже выполняется: `background step1 → human → hardware → background
step2 → background step3`. Фактический порядок проверять по строкам
`native-background STEP`, `native-scheduler HUMAN START` и
`native-hardware START` серверного журнала.

## 5. Ограничения

Это тест транспортного приоритета, а не утверждение, что аппаратные
прерывания уже подключены. У Task SYSTEM остаются права на события,
периодику, распределение RUN и подтверждение завершения. В этапе 5
новый CORE будет взаимодействовать с ним по штатному RPC.

Длительный инференс нельзя прервать в середине отдельного
`send_message()`. Подтверждённые приоритеты работают на границах
шагов; timeout здесь тоже проверяется на этих границах.
