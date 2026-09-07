# cat-agent

`cat-agent` is an experimental local LLM agent runtime for long-lived inference on edge hardware.

The project separates agent orchestration from model backends and human interfaces. Python owns runtime mechanics, scheduling, IPC, tools and state. The manager and agents operate inside prepared prompts/skills and decide how delegated work is carried out.

The current runtime supports three model backends behind the same CORE contract: LiteRT-LM, llama.cpp and OpenAI-compatible chat/completions endpoints.

## Current architecture

```text
                    interfaces
             TUI / Web / Voice
                       |
                 Unix socket IPC
                       |
                 +-----v-----+
                 |   CORE    |
                 | scheduler |
                 +-----+-----+
                       |
              AssistantManagerRuntime
                       |
              +--------+--------+
              |                 |
         LiteRT-LM          llama.cpp
              |                 |
        manager/agents     manager/agents
              +--------+--------+
                       |
          tools / tasks / MQTT events
                       |
              autonomous results
              human / Firebase
```

The CORE is headless and long-lived. It owns the scheduler, persistent task timers, system/hardware events, telemetry and active human session. Interfaces connect through `/run/cat-agent/core.sock` and do not depend on the selected model backend.

LiteRT-LM keeps resident manager/agent sessions directly in-process. The llama.cpp backend uses a local `llama-server` with fixed slots (`manager=0`, `agent=1`) and the same orchestration, scheduler and IPC boundary.

## Source layout

- `src/orchestration/` — shared manager/agent orchestration, prompts, skills, tools, tasks and system events.
- `src/agent_core/` — backend-independent scheduler, IPC, telemetry types and persistent notification delivery.
- `src/litert_agent/` — LiteRT-LM model adapter, TUI/Web/Voice frontends and compatibility imports.
- `src/openai_agent/` — OpenAI-compatible endpoint adapter and launcher.
- `src/llama_agent/` — llama.cpp model adapter and CORE launcher.
- `prompts/` — manager, agent and skill prompts.
- `config/` — runtime configuration such as MQTT first-active states.
- `tests/` — unit and integration tests for orchestration, CORE scheduling/IPC and model backend wiring.

## LiteRT-LM launch

The supplied launch scripts are deployment-specific and currently assume the project is installed in `/opt/cat-agent`, LiteRT-LM uses `/opt/litert-lm-venv`, and models are stored under `/storage/models/litertlm`.

Start the headless CORE with one of the configured model profiles:

```bash
cd /opt/cat-agent

# Gemma 4 E2B GPU build: GPU backend, fp32 activations,
# speculative decoding enabled, YNNPACK disabled.
./start_litert_agent.sh e2b

# Gemma 4 E4B: CPU backend, speculative decoding disabled,
# experimental YNNPACK enabled.
./start_litert_agent.sh e4b
```

## llama.cpp launch

The llama.cpp backend uses `CAT_AGENT_MODEL` and a local `llama-server` on port `9380`.

Start the model server:

```bash
cd /opt/cat-agent
export CAT_AGENT_MODEL=/storage/models/gemma-4-E4B-it-Q5_K_M.gguf
./start_llama_server.sh
```

Then start the same headless CORE contract with the llama.cpp backend:

```bash
cd /opt/cat-agent
export CAT_AGENT_MODEL=/storage/models/gemma-4-E4B-it-Q5_K_M.gguf
./start_llama_agent.sh
```

The llama CORE waits for `llama-server`, warms the manager and agent BASE prefixes, arms persistent task timers, starts MQTT event monitoring and only then exposes the CORE socket.

## Interfaces

After either CORE backend is ready, the same interfaces can connect. For example, the standalone TUI:

```bash
cd /opt/cat-agent
./start_tui.sh
```

The Web and Voice interfaces use the same CORE socket boundary. Autonomous results are routed to an active human interface or to Firebase when no human interface owns the session.

The CORE log is written to:

```text
/var/log/litertlm/cat-agent.log
```

## LiteRT-LM profiles

| Profile | Model | Backend | Speculative | YNNPACK |
| --- | --- | --- | --- | --- |
| `e2b` | `gemma-4-E2B-it-gpu.litertlm` | GPU | on | off |
| `e4b` | `gemma-4-E4B-it.litertlm` | CPU | off | on |

YNNPACK is enabled only for the E4B CPU profile. The E2B profile uses the dedicated GPU model build and is left on the GPU path.

## Tests

Run the test suite with the same Python environment used by the CORE:

```bash
cd /opt/cat-agent

PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
-m unittest discover -s tests -v
```

## Status

This is a working research project rather than a packaged general-purpose agent framework. Paths, model profiles and launchers currently reflect the target ARM64 deployment and are expected to evolve together with the agent architecture.


## 6-sept runtime update

See [the rollout and behavior notes](docs/6-sept.md) for the ten implementation
areas, configuration and Radxa verification steps. Restart CORE and its interfaces
together: the voice client now waits for explicit `completed` messages.

Requests have identities and are bound to their originating session. Human,
voice and notification manager contexts are isolated. Execution is cooperative:
a model call or command finishes before a higher-priority request can run.
LiteRT dialogue contexts use additional sessions on the existing manager engine;
llama.cpp uses full histories with cache comparison in the existing manager slot.
No additional model engine is loaded for these dialogues.

The Web frontend now binds to `127.0.0.1` by default. To expose it on the trusted
LAN, explicitly configure `CAT_AGENT_WEB_HOST` and either `CAT_AGENT_WEB_TOKEN`
or `CAT_AGENT_WEB_TRUSTED_NETWORK=1`. With a token, enter it through the Web
frontend's access-key button. Use TLS or an SSH tunnel outside a trusted LAN.

Notifications are persisted in `/var/lib/cat-agent/outbox.sqlite3`. The updated
TUI/Web acknowledge receipt; Firebase success means acceptance by the provider,
not confirmation that a phone displayed the notification. Delivery is at least
once, so a crash or partial Firebase failure can cause duplicate notifications.
Pending input events are still in-memory; they are not replayed after a restart.
