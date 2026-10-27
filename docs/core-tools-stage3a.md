# Native CORE — этап 3A: независимые агенты (проверка пула)

## Что сделано

Один LiteRT Engine теперь способен обслуживать раздельные BASE менеджера
и агентов. Для каждого агента создаётся собственный системный промпт,
собственный набор разрешённых нативных инструментов, BASE и рабочий
Conversation. Прогрев всех BASE выполняется последовательно. Shell во
время прогрева отключён, затем включается только для разрешённых
инструментов.

Новый `orchestration_tools.agents.AgentPool` управляет состояниями
`FREE/RUNNING/WAITING/FAILED`. В этой итерации агенту даётся **одно**
задание через `send_message()`, по завершении рабочий диалог
клонируется заново из ЕГО BASE. Даже после ошибки агент не продолжает
повреждённую историю. Если восстановление невозможно, агент остаётся
`FAILED`.

Все обращения менеджера и агентов проходят через **один** поток
исполнителя IPC. Мы не делаем вложенный `send_message()` в callback
нативного инструмента и не запускаем параллельные декодирования.

## Конфигурация

Созданы:

- `prompts_tools/agent_1.md` — новый промпт, **без `/work#`**.
- `config/native_agents.example.json` — пример задания имени, пути
  к промпту и индивидуального списка разрешённых инструментов.

Поля одного агента:

```json
{"id": "agent_1", "prompt": "/opt/cat-agent/prompts_tools/agent_1.md", "tools": ["shell"]}
```

Конфигурация строго проверяется до загрузки модели. Список допускает
только зарегистрированные инструменты. Для `shell` обязательно явно
включить `--enable-shell`, иначе запуск с таким конфигом отвергается.
У каждого агента могут быть собственные права, независимо от прав
менеджера.

Это экспериментальный пример, не окончательное распределение навыков.

## Проверка на WSL

Остановить старый экспериментальный процесс на том же сокете, чтобы
не загружать вторую копию модели.

```bash
cd /opt/cat-agent
git pull --ff-only

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_agent_core_tools*.py' -v

PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m unittest discover -s tests -p 'test_orchestration_tools_agents_runtime.py' -v

./start_litert_agent_tools.sh \
  --socket /tmp/cat-agent-litert-tools-test.sock \
  --model /storage/models/litert/gemma-4-E4B-it.litertlm \
  --system-file /opt/prompt/sys_prompt_manager.md \
  --agent-config /opt/cat-agent/config/native_agents.example.json \
  --enable-shell \
  --trace-native-json
```

Другой терминал:

```bash
python3 - <<'PY'
import json, socket
p = '/tmp/cat-agent-litert-tools-test.sock'
with socket.socket(socket.AF_UNIX) as s:
    s.connect(p)
    with s.makefile('r', encoding='utf-8') as rd:
        def exchange(msg):
            s.sendall((json.dumps(msg, ensure_ascii=False) + '\n').encode())
            data = json.loads(rd.readline())
            print(data)
            return data

        exchange({'type': 'acquire', 'client': 'agent-test'})
        exchange({'type': 'snapshot'})
        for text in ('Выполни shell pwd и сообщи результат',
                     'Выполни shell ls и перечисли файлы'):
            accepted = exchange({'type': 'agent_task', 'agent': 'agent_1',
                                 'text': text})
            if accepted['type'] != 'accepted':
                raise RuntimeError(accepted)
            while True:
                data = json.loads(rd.readline())
                print(data)
                if data['type'] == 'completed':
                    break
        exchange({'type': 'release'})
PY
```

Ожидается, что `snapshot.status.agents` содержит
`agent_1, FREE, base_kv_tokens=...`, агент дважды использует shell,
а между заданиями его рабочий KV сбрасывается к BASE.

## Границы этапа

`agent_task` — **временный диагностический IPC-маршрут** только
для владельца сеанса. Это ещё не делегирование менеджером. Автономные
TASK, постоянная очередь заданий, приоритеты, отмена, агентские tool
callbacks для делегирования и контроль числа внутренних вызовов
`send_message()` пока не реализованы.

Следующий этап 3B: создать `delegate_task` как **нативный инструмент
менеджера**, возвращающий только факт постановки поручения. После
завершения текущего хода менеджера CORE должен отдельно исполнить
поручение и передать фактический результат менеджеру. Это позволяет
избежать недоказанного вложенного инференса одного Engine.
