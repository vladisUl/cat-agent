# cat-agent

`cat-agent` is an experimental local LLM agent runtime for long-lived inference on edge hardware.

The project separates agent orchestration from model backends and human interfaces. Python owns runtime mechanics, scheduling, IPC, tools and state. The manager and agents operate inside prepared prompts/skills and decide how delegated work is carried out.

The current primary runtime is LiteRT-LM on Linux ARM64. A llama.cpp/OpenAI-compatible backend is also kept in the repository as an alternate runtime.

## Current architecture

```text
              human interface
                   TUI
                    |
              Unix socket IPC
                    |
              +-----v-----+
              |   CORE    |
              | scheduler |
              +-----+-----+
                    |
              manager model
                    |
                 agents
                    |
        tools / tasks / system events
```

The LiteRT-LM CORE is headless and long-lived. It owns the resident manager and agent model engines, scheduler, persistent task timers, system/hardware events, telemetry and the active human session.

The current human interface is a standalone curses TUI connected to the CORE through `/run/cat-agent/core.sock`. The CORE is intentionally independent of the TUI so interfaces can connect and disconnect without reloading the models. Additional interfaces can be added on the same boundary; Telegram integration is not implemented yet.

## Source layout

- `src/orchestration/` — shared manager/agent orchestration, prompts, skills, tools, tasks and system events.
- `src/litert_agent/` — LiteRT-LM runtime, resident model clients, CORE scheduler/server, IPC client and standalone TUI.
- `src/llama_agent/` — alternate llama.cpp/OpenAI-compatible interactive runtime.
- `prompts/` — manager, agent and skill prompts.
- `tests/` — unit and integration tests for orchestration, CORE scheduling/IPC and LiteRT runtime wiring.

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

Then start the standalone TUI in another terminal:

```bash
cd /opt/cat-agent
./start_tui.sh
```

The CORE log is written to:

```text
/var/log/litertlm/cat-agent.log
```

The model/backend policy is kept in `start_litert_agent.sh`; `src/litert_agent/runtime.py` reads the exported LiteRT-specific environment variables and builds the resident engines.

## LiteRT-LM profiles

| Profile | Model | Backend | Speculative | YNNPACK |
| --- | --- | --- | --- | --- |
| `e2b` | `gemma-4-E2B-it-gpu.litertlm` | GPU | on | off |
| `e4b` | `gemma-4-E4B-it.litertlm` | CPU | off | on |

YNNPACK is enabled only for the E4B CPU profile. The E2B profile uses the dedicated GPU model build and is left on the GPU path.

## llama.cpp backend

The repository also contains a separate llama.cpp/OpenAI-compatible path. It uses `CAT_AGENT_MODEL` and a local OpenAI-compatible server on port `9380`:

```bash
export CAT_AGENT_MODEL=/path/to/model.gguf
./start_llama_server.sh
```

In another terminal:

```bash
export CAT_AGENT_MODEL=/path/to/model.gguf
./start_llama_agent.sh
```

This path is currently a direct interactive runtime and does not use the LiteRT-LM CORE/TUI IPC split.

## Tests

Run the test suite with the same Python environment used by LiteRT-LM:

```bash
cd /opt/cat-agent

PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
-m unittest discover -s tests -v
```

## Status

This is a working research project rather than a packaged general-purpose agent framework. Paths, model profiles and launchers currently reflect the target ARM64 deployment and are expected to evolve together with the agent architecture.
