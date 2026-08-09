# cat-agent 0.1

Новая ветка архитектуры: один manager, три нейтральных agent-контейнера и внешняя база skill-prompts.

## Запуск llama-server

Для Gemma 4 в этом проекте reasoning должен быть отключён на стороне llama.cpp:

```bash
cd /opt/llama.cpp
./build-vulkan/bin/llama-server \
  -m /opt/llama.cpp/models/gemma-4/gemma-4-E4B-it-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 9380 \
  --parallel 1 \
  --device none \
  --jinja \
  --reasoning off \
  --reasoning-budget 0
```

## Запуск cat-agent

```bash
cd /opt/cat-agent
CAT_AGENT_API_BASE_URL=http://127.0.0.1:9380/v1 \
CAT_AGENT_MODEL='/opt/llama.cpp/models/gemma-4/gemma-4-E4B-it-Q4_K_M.gguf' \
python3 -m cat_agent.main
```

По умолчанию `WORKSPACE=/opt/model`, prompt-файлы берутся из `./prompts`.

## Протокол manager

- `DELEGATE skill1,skill2` + задача
- `CONTINUE agentN` + дополнительные данные после `EVENT NEED`
- `ASK` + вопрос пользователю
- `WAIT`
- `REPLY` + итоговый ответ

## Протокол agent

- одна Linux-команда в одной строке;
- `DONE` + итог;
- `NEED` + недостающие данные.
