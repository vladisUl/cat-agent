# MCP tools

MCP is an optional additional tool mechanism. `/work#`, shell/CLI, MQTT,
`read_pic` and TASK SYSTEM retain their existing semantics.

## Install and check

On Radxa, in the same venv used by CORE:

```sh
cd /opt/cat-agent
git pull --ff-only
/opt/litert-lm-venv/bin/python3 -m pip install -e '.[mcp]'
PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
  -m unittest discover -s tests -v
```

The optional extra pins the tested official Python MCP SDK to 2.2.0 and includes
JSON Schema validation. No SDK import, thread, subprocess or connection occurs
when there are no enabled servers. An unavailable optional dependency with MCP
enabled is logged; ordinary CORE tools continue working.

## Configuration

The `mcp` section is optional. An old YAML without it is valid. Configuration is
read from `CAT_AGENT_CONFIG`, or the repository's `cat-agent.yaml`. Examples:

```yaml
mcp:
  servers:
    - name: demo
      enabled: true
      manager: true
      description: Небольшой demo MCP, доступный Гене напрямую и агентам.
      transport: stdio
      command: /opt/litert-lm-venv/bin/python3
      args: [/opt/cat-agent/tests/fixtures/mcp_server.py]
      connect_timeout_seconds: 10
      call_timeout_seconds: 30
      reconnect_delay_seconds: 5

    - name: remote
      enabled: false
      manager: false
      description: Удалённый специализированный MCP для агентских заданий.
      transport: streamable_http
      url: http://127.0.0.1:8790/mcp
      connect_timeout_seconds: 10
      call_timeout_seconds: 30
```

`enabled: false` does not connect or spawn a process. `manager` defaults to `true`
for backward compatibility. With `manager: true`, full tool schemas are added to
Manager BASE and the manager may call those leaf tools directly. With
`manager: false`, Manager BASE contains only the short `mcp:<server>` capability
and `description`; the full frozen schemas are supplied only to an agent assigned
that capability. Names must be unique and
match `[a-z][a-z0-9_-]*`. Tool names must contain only letters, digits, `_`, `-`
and `.`. Conflicting definitions from one server are rejected, never overwritten.

The stdio command is an executable, `args` is an argument array; it is not passed
through a shell. Use absolute paths for the executable and script. A stdio server
writes protocol messages to stdout and diagnostics to stderr.

Optional environment references:

```yaml
# Inside a stdio server entry:
env:
  SERVICE_API_KEY: HOME_SERVICE_API_KEY

# Inside an HTTP server entry:
headers_env:
  Authorization: HOME_MCP_AUTHORIZATION
```

These values are names of environment variables in CORE's environment. The HTTP
variable must contain the complete header value, including `Bearer ` if needed.
Missing referenced variables make that server unavailable. Secret values are not
included in the tool catalog or adapter diagnostics. HTTP uses the SDK/httpx2
standard proxy environment; a SOCKS proxy requires its optional SOCKS dependency.

## Fixed catalog and execution

Before building/warming BASE, every enabled server gets one bounded discovery
attempt (servers are contacted concurrently). The SDK obtains capabilities and
all pages of `tools/list`. The catalog is then frozen for the lifetime of CORE.
Descriptions and schemas are frozen as well as names. Each enabled server also
produces a generated diagnostic snapshot in `/opt/cat-agent/mcp/<server>.txt`.
The snapshot is derived from the frozen `tools/list` result and is never an input
or configuration source.

Manager sees one short capability `mcp:<server>` for every discovered server.
Only servers configured with `manager: true` additionally put their full leaf
tool schemas into Manager BASE and grant direct execution. Assigning
`mcp:<server>` to an agent expands that capability to the full frozen tool set
for that server and grants only those MCP leaf calls. Existing saved tasks that
name a full leaf tool such as `mcp:demo:echo` remain valid.

If a server was unreachable at startup, its tools are absent until CORE restart.
A later connection does not add them. If a discovered server goes down, its tools
remain in the catalog; calls return an unavailable error until reconnection.
Reconnect may read `tools/list` to verify the live definitions, but never modifies
the frozen catalog. If a tool disappears or its input schema changes, calling it
returns `tool_changed` and requires a CORE restart. Description changes alone do
not replace the frozen description.

The model requests one action:

```text
/work#mcp:demo:echo {"text":"Проверка MCP"}
```

A shared dispatcher handles manager and agent commands before shell parsing.
It validates the JSON object, tool assignment and startup input schema. The
SDK executes `tools/call` using the server's original tool name. The model gets
`MCP_RESULT` followed by a normalized payload with explicit `isError` and ordered
`content` blocks. Text, media data, resource URIs and block annotations are kept.
Top-level SDK metadata is not included. `structuredContent` contains only data
not already represented by JSON text blocks: equal object fields at the same
path are omitted, while extra/conflicting fields are kept. Arrays are compared
as whole ordered values, never shortened/reindexed. Comparison ignores JSON
whitespace/key ordering but distinguishes booleans, numbers, strings and null.
The SDK's single-key `result` wrapper is also omitted when it exactly repeats
one text value, the parsed JSON value, or an entire ordered sequence of text
blocks representing list items. Arbitrary prose is not interpreted or
summarized, and JSON inside embedded resources is not treated as root data. This first version preserves
content blocks as JSON; it does not turn MCP image/audio blocks into model-native
media inputs, fetch resource links, or add resources/prompts/sampling/elicitation.
The existing `read_pic` image path is independent and unchanged.

A tool's `isError` result does not disconnect the server. Connection failures and
timeouts trigger reconnection, but **never automatic replay of tools/call**.
`outcome_unknown` means the external action may have happened. Repeating the same
name/JSON arguments is refused within the current execution context. The guard
is not a durable exactly-once guarantee across new sessions or CORE restarts.

MCP calls consume the existing model step budget and have a bounded synchronous
wait, just like other tool calls; the CORE scheduler is not rewritten.

## TASK and multiple COREs

Prefer assigning the server capability when an agent may choose among that MCP
server's tools:

```text
/work#task_timer.sh 0 mcp:demo -- "Повтори текст Проверка MCP"
/work#task_timer.sh 60 mcp:demo -- "Повтори текст Проверка MCP"
```

A full leaf name is still accepted for narrow or previously saved tasks:

```text
/work#task_timer.sh 60 mcp:demo:echo -- "Повтори через инструмент echo текст Проверка MCP"
```

All manager/TASK `require()` paths resolve through the common catalog, including
saved tasks loaded after restart. Existing executor and RUN policies are unchanged.
Configure the same required MCP tools in both COREs for `executor=auto`; missing
MCP tools do not cause a started RUN to move to another CORE.

Each CORE owns its client pool, shared by its manager contexts and agents. Two
COREs create two instances of each enabled stdio server. Use Streamable HTTP to
connect both COREs to one external server process. Closing a CORE closes only its
own sessions and stdio children.

## Manual smoke test

Enable the `demo` entry above and restart the selected CORE using its usual
launcher/service. There is no need to restart TASK SYSTEM. Check the log for:

```text
MCP server=demo state=ready reason=none
MCP catalog frozen: servers=1 tools=4
```

In TUI/Web ask: `Через mcp:demo:echo повтори текст Проверка MCP`.
Then ask for a one-shot TASK using the same tool. Its actual result must return
through the normal manager/agent path, without invoking a shell command named mcp.

For a real HTTP server, start in a separate terminal:

```sh
/opt/litert-lm-venv/bin/python3 /opt/cat-agent/tests/fixtures/mcp_server.py --port 8790
```

Enable `remote`, restart CORE and ask it to use `mcp:remote:echo`. Stop the demo
HTTP server: a call fails while CORE and shell/MQTT continue to work. Restart the
server: subsequent calls recover automatically after the reconnect delay, with
the same catalog and BASE.

Focused checks (both real SDK transports, no LLM required):

```sh
cd /opt/cat-agent
PYTHONPATH=/opt/cat-agent/src /opt/litert-lm-venv/bin/python3 \
  -m unittest discover -s tests -p 'test_mcp*.py' -v
```
