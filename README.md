# cat-agent

`cat-agent` is an experimental long-lived LLM agent runtime built for a real
edge-device deployment rather than as a generic chat wrapper.

The project separates model inference, orchestration, persistent autonomous
work, tools and human interfaces. A MANAGER handles the dialogue and decides
whether to answer directly, call a tool or delegate work. AGENT workers execute
bounded tasks with an explicitly assigned capability set. Model backends are
replaceable; the orchestration and task semantics are shared.

The current deployment is centered on ARM64 hardware and uses fixed paths such
as `/opt/cat-agent` and `/opt/model`. Those paths are part of the deployment
today, not a portability promise.

## Architecture

```text
                         human interfaces
                    TUI / Web / Voice
                           |
                    CORE selection
                 auto: OpenAI -> LiteRT
                      /           \
                     /             \
        /run/cat-agent/openai.sock  /run/cat-agent/litert.sock
                   |                       |
            +------v------+         +------v------+
            | OpenAI CORE |         | LiteRT CORE |
            | MANAGER     |         | MANAGER     |
            | AGENT pool  |         | AGENT pool  |
            +------+------+         +------+------+
                   |                       |
                   +-----------+-----------+
                               |
                    shared orchestration
                       prompts / tools
                               |
                    +----------+----------+
                    |                     |
               tools/*.md                MCP
          canonical TOOL blocks        optional
                    |
                    v
             /opt/model/<tool>.sh
             internal tool scenario
                    |
             real executables +
              internal opcodes
                                |
                       /opt/model/data
                                |
              +-----------------+-----------------+
              |
      /run/cat-agent/tasks.sock
              |
        +-----v------+
        | Task SYSTEM|
        | SQLite     |
        | timers     |
        | MQTT runs  |
        | CORE READY |
        +------------+
```

LiteRT and OpenAI-compatible COREs may run at the same time. Human interfaces
can explicitly select one CORE or ask the launcher to select the preferred
READY backend. Persistent autonomous work is owned by a separate Task SYSTEM,
so tasks do not belong to the CORE through which they were created.

An experimental llama.cpp backend is also present. It shares the orchestration
layer, but it is not currently an automatic executor in the shared Task SYSTEM
policy.

## MANAGER and AGENT

The MANAGER owns the interactive dialogue. It can:

- answer the user;
- execute tools that are granted directly to MANAGER;
- create and manage autonomous TASKs;
- delegate work to an AGENT with a specific set of tools;
- continue from actual runtime results instead of assuming that an external
  action succeeded.

AGENT workers receive only the capabilities assigned to the task. Their
execution is bounded by the same step budget and runtime rules as interactive
work.

The orchestration follows a strict turn boundary: every model call must have a
fresh user/system input. An AGENT step is one model TICK -> TOCK and stops at
the next boundary. Tool results therefore become explicit new input instead of
being hidden side effects inside a model turn.

## Multiple COREs and selection

The standard LiteRT and OpenAI CORE sockets are:

```text
/run/cat-agent/litert.sock
/run/cat-agent/openai.sock
```

Interface launchers accept an optional explicit backend:

```bash
./start_tui.sh litert
./start_web.sh openai
./start_voice.sh litert
```

With no argument, the launcher asks Task SYSTEM for CORE readiness and selects
OpenAI first, then LiteRT:

```bash
./start_tui.sh
./start_web.sh
./start_voice.sh
```

`CAT_AGENT_CORE_SOCKET` remains the low-level override for tests and
administrative use.

See [docs/core-selection.md](docs/core-selection.md) for socket ownership,
locking and selection details.

## Shared Task SYSTEM

Task SYSTEM is a model-free service and the single owner of persistent tasks,
timers, MQTT bindings and queued runs. Its default socket is:

```text
/run/cat-agent/tasks.sock
```

State is stored in SQLite under `/var/lib/cat-agent`. A saved TASK has an
executor policy:

- `auto` — choose OpenAI when it is READY, otherwise LiteRT; if neither is
  READY the run remains pending;
- `openai` — administrative pinning to the OpenAI CORE;
- `litert` — administrative pinning to the LiteRT CORE.

A RUN that has already started is never silently moved to another CORE. If a
CORE dies after execution began, the outcome may be unknown and that run is not
automatically replayed: external side effects may already have happened.

Useful administrative commands:

```bash
./task_system.sh list
./task_system.sh cores
./task_system.sh runs
./task_system.sh run 1
./task_system.sh stop 1
./task_system.sh start 1
./task_system.sh period 1 120
./task_system.sh executor 1 auto
./task_system.sh delete 1
```

See [docs/task-system.md](docs/task-system.md) for persistence, queue states,
heartbeats, executor selection, MQTT activation and restart semantics.

## Tool catalog

Each CORE builds one immutable tool catalog before its model BASE is prepared.
Local model-facing tool definitions live only in `tools/*.md`. Enabled MCP
servers may add discovered tools to the same frozen runtime catalog.

A local file is the canonical definition presented to the model. Its contents
are not paraphrased or regenerated by Python. For example:

```text
**[TOOL prognoz]**

**name:** prognoz
**code:** /work#prognoz.sh
**description:** прогноз погоды
**manager:** true

**[/TOOL]**
```

At CORE startup:

- every `tools/*.md` file is validated;
- tools with `manager: true` are copied verbatim into MANAGER's system prompt
  after `СПИСОК TOOLS`;
- an AGENT receives the same verbatim TOOL blocks for the tools assigned to its
  task;
- `prompts/mqtt.md` is copied after the MQTT topic marker;
- the `skills/` directory is currently not a runtime tool source.

The catalog stays frozen for the lifetime of a CORE. Adding or changing a tool
definition therefore requires a CORE restart so the model BASE and runtime
authorization set stay consistent.

Some local tools use a workspace scenario, normally:

```text
/opt/model/<name>.sh
```

Despite the `.sh` suffix, these scenario files are not necessarily Bash.
They are line-oriented runtime scenarios. Internal operations such as
`read_pic.sh`, `mqtt_sub.sh`, `mqtt_pub.sh` and `cyclic_process` are
handled by cat-agent; other first tokens resolve to executable files in the
workspace root.

A tool command may carry single-line input after the scenario name. The input is
passed to the first scenario step, then textual output flows to the next step.
The final textual output is returned to the model; an empty final output means
successful silent completion. DATA path traversal and symlink escape are
rejected.

## DATA workspace

Temporary application data is rooted under:

```text
/opt/model/data
```

The deployment may mount this directory as tmpfs. Skill scenarios use logical
DATA paths rather than arbitrary Linux paths. With the normal workspace both of
these refer to the same file:

```text
camera/test.png
/camera/test.png
    -> /opt/model/data/camera/test.png
```

Subdirectories are supported. `..` traversal and symlink escape outside DATA
are rejected.

Executable programs themselves remain in `/opt/model`; DATA is for runtime
artifacts, not executable code.

## Image input

`read_pic.sh` attaches PNG or JPEG data from the DATA workspace to the current
model context:

```text
/work#read_pic.sh camera/test.png
```

It is an internal command; no Linux executable named `read_pic.sh` is needed.

The OpenAI-compatible adapter sends the image using a data URL. LiteRT-LM
migrates the active text session into a multimodal Conversation on the first
image and keeps subsequent dialogue in that context. The current llama.cpp
adapter reports image input as unsupported.

See [docs/read-pic.md](docs/read-pic.md) for format, size and backend details.

## MCP

MCP support is optional:

```bash
/opt/litert-lm-venv/bin/python3 -m pip install -e '.[mcp]'
```

Enabled stdio and Streamable HTTP servers are discovered before the model BASE
is finalized. The resulting catalog is frozen until CORE restart.

MCP servers use the same visibility idea as dynamic skills:

- `manager: true` — MANAGER receives full leaf schemas and may call them
  directly;
- `manager: false` — MANAGER sees only the server capability
  `mcp:<server>`; an assigned AGENT receives the frozen leaf tools.

Tool results pass through a common normalizer. It preserves `isError` and
content blocks while avoiding a second semantically duplicate
`structuredContent` representation.

Connection failure and timeout handling never automatically replays
`tools/call`. When an external side effect may already have happened, the
runtime reports an uncertain outcome instead of pretending the call was safely
retryable.

See [docs/mcp.md](docs/mcp.md) for configuration and exact result semantics.

## Model backends

### LiteRT-LM

LiteRT runs in-process and is configured through the `litert` section of
`cat-agent.yaml`. The active profile selects model path, CPU/GPU backend,
thread count, speculative decoding and YNNPACK settings.

Start it with:

```bash
cd /opt/cat-agent
./start_litert_agent.sh
```

The profile is selected by `litert.active_profile` in YAML; the launcher no
longer accepts a profile argument.

LiteRT keeps resident model sessions and prepares reusable manager/agent BASE
contexts so ordinary requests do not pay the full system-prompt prefill on each
turn.

### OpenAI-compatible

The OpenAI adapter works with Chat Completions-compatible endpoints and is
configured through the `openai` section of `cat-agent.yaml`.

```bash
cd /opt/cat-agent
./start_openai_agent.sh
```

Secrets may be supplied through `.env.openai`; ordinary endpoint/model
configuration belongs in YAML.

The current deployment can use Ollama readiness and account-usage checks for
automatic TASK executor selection. Readiness affects `executor=auto`; it does
not guarantee that a subsequent generation request cannot fail.

### llama.cpp

The llama.cpp adapter remains available for experiments with a local
`llama-server`:

```bash
export CAT_AGENT_MODEL=/path/to/model.gguf
./start_llama_server.sh
./start_llama_agent.sh
```

It shares the orchestration layer but is not currently selected by the standard
`litert|openai` interface launcher or by automatic persistent TASK execution.

## Configuration

The main configuration file is `cat-agent.yaml`. It currently contains
sections for:

- LiteRT model profiles;
- OpenAI-compatible profiles;
- manager/agent limits;
- workspace and timeouts;
- Web and Voice settings;
- Firebase notifications;
- optional MCP servers.

A commented/reference configuration is kept in `cat-agent.yaml.doc`.

## Starting the deployment

For manual startup, start Task SYSTEM first and then one or both primary COREs:

```bash
cd /opt/cat-agent

./start_task_system.sh
./start_litert_agent.sh
./start_openai_agent.sh
```

Run each long-lived process separately or under a service manager.

The repository also contains systemd units and convenience wrappers. On the
target installation:

```bash
./all_start.sh
./all_stop.sh
```

manage:

```text
cat-agent-task-system.service
cat-agent-litert.service
cat-agent-openai.service
```

After a CORE is READY, connect an interface:

```bash
./start_tui.sh
./start_web.sh
./start_voice.sh
```

or choose a backend explicitly:

```bash
./start_tui.sh litert
./start_web.sh openai
```

## Source layout

- `src/orchestration/` — MANAGER/AGENT logic, tool catalog/dispatch, MCP, MQTT,
  image handling and shared runtime contracts.
- `src/agent_core/` — CORE IPC, scheduling, telemetry and notification
  infrastructure.
- `src/task_system/` — persistent TASK registry, timers, run queue and CORE
  readiness service.
- `src/litert_agent/` — LiteRT-LM backend plus shared TUI/Web/Voice frontend
  implementations.
- `src/openai_agent/` — OpenAI-compatible backend.
- `src/llama_agent/` — llama.cpp backend.
- `prompts/` — authored MANAGER/AGENT prompts and MQTT context.
- `tools/` — canonical local TOOL definitions loaded at CORE startup.
- `skills/` — currently unused.
- `docs/` — design and deployment notes for individual subsystems.
- `systemd/` — service units for the primary deployment.
- `tests/` — orchestration, runtime, IPC, Task SYSTEM, MCP and backend tests.

## Tests

Run the full suite with the Python environment used by the deployment:

```bash
cd /opt/cat-agent
git pull --ff-only

PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
-m unittest discover -s tests -v
```

The test count is intentionally not documented here; the suite grows with the
runtime.

## Documentation

Subsystem notes are kept separate from this overview:

- [CORE selection](docs/core-selection.md)
- [Task SYSTEM](docs/task-system.md)
- [Tool scenarios / legacy skill scripts](docs/skill-scripts.md)
- [Image input](docs/read-pic.md)
- [MCP](docs/mcp.md)
- [LiteRT session warmup](docs/7-sept-session-warmup.md)
- [Earlier September runtime rollout](docs/6-sept.md)

## Project status

This is a working research system under active development. It already runs
interactive dialogue, autonomous scheduled/event-driven work, multiple model
backends, persistent task routing, local tools, multimodal image input and optional MCP integration.

It is not a packaged general-purpose agent framework. The repository follows
the needs of the actual deployment, and architectural contracts are preferred
over compatibility with arbitrary environments.
