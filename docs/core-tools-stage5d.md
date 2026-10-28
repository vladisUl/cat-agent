# Native CORE — этап 5D: границы отказа и запрет повторных side effects

Этап следует за проверенным на Gemma 4 E4B маршрутом 5C:
TASK SYSTEM → AGENT → QUERY MANAGER → SQLite inbox → IPC ACK.
Модель, Native Tool Calling и прежний CORE не меняются.

## Контракт

- Единственный владелец TASK/RUN, lease, event, таймера и постоянного
  QUERY inbox — TASK SYSTEM.
- Native CORE использует **один** serialized LiteRT Engine. Предыдущий
  вызов модели и side-effecting native tools автоматически никогда не
  повторяются в рамках RUN.
- После `begin` без полученного ответа CORE не знает, был ли принят
  RUN на сервере. Он **не выполняет модель**, удерживает token до
  подтверждения `finish` и повторяет только RPC. Если сервер так и не
  обработал `begin`, `finish(offered)` возвращает предложение в `pending`.
  Если `begin` успел сохраниться — `finish(running)` помечает
  выполнение `interrupted`. RUN не остаётся навечно в `running`
  при временной потере ответа.
- Потерянный ответ `valid` **перед первым вызовом AGENT** также означает
  отсутствие разрешения на side effects; модель не вызывается, RUN
  завершается как `interrupted` (если сервер уже зарегистрировал
  `running`). Если сервер явно ответил `false`, RUN отменяется.
- **Между AGENT и QUERY MANAGER** временный отказ `valid` не
  трактуется как отзыв разрешения. Результат AGENT хранится в RAM
  native CORE; после восстановления RPC повторяется только
  `worker.valid`, а затем запускается отдельный query-manager.
  Последний может запуститься только после явного `valid=True`.
  При `valid=False` второй модельный вызов отменяется.
- В момент graceful shutdown отложенные или ожидающие
  подтверждения QUERY-переходы не запускаются. После остановки
  сериализованного scheduler текущий RUN подтверждается как
  `interrupted`, без автоматического повторного запуска AGENT.
- После **аварийного** прекращения процесса CORE Task SYSTEM помечает уже
  начатый RUN `interrupted`, когда старый worker process identity мёртв.
  Только `offered`, то есть не начатые RUN, выдаются заново.
- Если QUERY уже сохранён в SQLite (`publish_result=True`),
  уведомление остаётся до пользовательского ACK даже в случае гибели
  worker до `finish`. RUN при этом может быть `interrupted`: это не
  означает, что сохранённый отчёт исчез или модель должна повториться.

## Границы

Гарантировать сохранность результата AGENT **до первого успешного
SQLite publish** при `SIGKILL` нельзя: в native CORE нет своего дискового
outbox. Он намеренно не вводился. Повторное выполнение инструментов
вместо восстановления результата недопустимо. Такую аварию диагностируем
по состоянию `interrupted`.

Нет preemption внутри `send_message`: остановка производится на
кооперативной границе вызовов или после окончания текущей инференции.
`SIGKILL` обрывает процесс без graceful cleanup; TASK SYSTEM
определяет потерю по process identity.

## Автоматические проверки

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

Новые тесты:

- `test_agent_core_tools_stage5d.py` — 8 сценариев отказа
  (RPC `pull/begin/valid`, второй модельный этап, отзыв RUN, shutdown).
- `test_task_system_stage5d.py` — 3 сценария гибели worker
  (running → interrupted, offered → pending, сохранённый QUERY inbox).
- В `test_agent_core_tools_task_worker.py` обновлено ожидание при
  неопределённом результате `BEGIN`.

## Ручная проверка на изолированной WSL-службе

Использовать **только** тестовые TASK SYSTEM/SQLite/IPC из
`docs/core-tools-stage5a.md` и `docs/core-tools-stage5c.md`.

A. Короткий обрыв TASK SYSTEM между AGENT и QUERY MANAGER тестируется
детерминированной инъекцией `TaskStoreError` из unit-test.
Подменять реальный сервер в случайный момент не требуется: это не
позволяет воспроизводимо попасть между двумя инференциями.

B. Чтобы проверить аварийное завершение CORE без side-effects replay:

1. Создать новый `method=query` RUN с безопасной командой `pwd`;
   зафиксировать `run_id` и PID native CORE. Не использовать
   задания, которые меняют файлы или управляют оборудованием.
2. Убить тестовый CORE через `SIGKILL` в момент, когда RUN уже
   `running`. Операционная система уничтожит его вместе с LiteRT.
3. Оставить тот же тестовый TASK SYSTEM/SQLite; `system.runs` должен
   показать `interrupted` после следующего `tick`.
4. Перезапустить CORE. Проверить, что старый `run_id` **не**
   выдан повторно и не возник второй вызов `shell`. Новый ручной
   `run_task` создаёт **другой run_id** и является отдельным действием.

Проверка `SIGKILL` рискованна для среды WSL/Gemma и не является
необходимой для работы этого патча. Детерминированные unit-тесты
устанавливают state-machine инварианты без принудительной гибели
реального процесса.

C. Проверка 5C должна оставаться зелёной: QUERY STORED,
`notifications.list` и `task_query_ack` проходят как раньше.

5D завершаем только после запуска всех трёх групп тестов на WSL и
анализа фактических результатов. Изменения, внесённые в GitHub,
сами по себе не означают, что 5D испытан на WSL.
