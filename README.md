# cat-agent

Local manager/agent runtime for an OpenAI-compatible `llama-server` backend.

The manager selects prepared skills and delegates work to neutral agent containers. Python owns runtime mechanics, command execution, state and safety checks; the model owns strategy inside the supplied prompts and skills.

## Model slots

The current runtime is sequential at the model level and uses two fixed llama-server slots:

- slot `0` — manager
- slot `1` — shared agent execution slot

Each role therefore keeps its own KV/prompt cache while control switches between manager and agent calls. The server must expose at least two slots.

Example llama-server startup:

```bash
llama-server \
  -m /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 9380 \
  --parallel 2 \
  --device none \
  --jinja \
  --reasoning off \
  --reasoning-budget 0 \
  --cache-prompt \
  --cache-ram 0 \
  --ctx-checkpoints 0
```

## Warmup

After `llama-server` is ready, the two persistent role slots can be prefilled before the interactive application starts:

```bash
python3 -m cat_agent.warmup_manager
python3 -m cat_agent.warmup_agent
```

The manager warmup evaluates its normal bootstrap in slot `0`.

The agent warmup evaluates only the immutable MQTT agent prefix in slot `1`. It asks llama-server to apply the active chat template to two tasks with deliberately different contents, finds their token-level common prefix, and submits exactly that prefix to `/completion` with `n_predict=0`. No fake TASK or generated answer remains after the warmup boundary.

Real agent conversations use complete stable bootstrap turns followed by TASK as a separate user turn. Changing only TASK therefore does not rewrite the large workspace/skill/context prefix.

## Run

Python 3.10 or newer is required. The project has no Python package dependencies.

Defaults are defined in `cat_agent/config.py`. The important model settings can be overridden with environment variables:

```bash
export CAT_AGENT_API_BASE_URL=http://127.0.0.1:9380/v1
export CAT_AGENT_MODEL=/path/to/model.gguf
export CAT_AGENT_WORKSPACE=/opt/model
```

Start the application:

```bash
python3 -m cat_agent.main
```

The MQTT skill uses the system `mosquitto_sub` and `mosquitto_pub` commands when that skill is selected.

## Prompts and skills

- `prompts/sys_prompt_manager.txt` — manager system prompt
- `prompts/sys_prompt_agent_N.txt` — neutral agent system prompts
- `prompts/prompt_base.txt` — available skill definitions
- `prompts/<skill>.txt` — optional environment-specific context for a skill

Manager prompts receive the skill catalog only. A delegated agent receives the full selected skill prompt plus its optional context.

## Tests

```bash
python3 -m unittest discover -s tests -v
```
