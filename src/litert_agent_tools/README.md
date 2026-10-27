# Native tools experiment (LiteRT Conversation)

This package is deliberately separate from src/litert_agent, the OpenAI
client, MCP, /work#, and TASK SYSTEM. None of their code or prompts is modified.

## Scope

- Four native zero-argument Python tools registered in Conversation(tools=...):
  outdoor temperature/humidity, home temperature/humidity, aquarium temperature
  and freezer temperature. Topics are the existing entries in prompts/mqtt.md.
- MQTT tools are read-only; each invokes mosquitto_sub once. They return JSON
  with ok=false on errors.
- Optional native shell(command: str) is available with --enable-shell. It
  executes Bash with the current OS user's full permissions; shell is disabled
  by default. There is no /work# intermediary.
- The system prompt is supplied by the operator via --system-file.
- One BASE Conversation plus a disposable working clone. /new closes the old
  conversation and clones BASE, so its history and KV are independent.

## Native shell tool (explicit opt-in)

The native function is `shell(command: str) -> dict` and is registered in
`Conversation(tools=...)` alongside MQTT. LiteRT-LM 0.18.0 constructs
a tool schema from the Python signature and invokes it automatically.

```sh
cd /opt/cat-agent
PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m litert_agent_tools \
  --model /storage/models/litert/gemma-4-E4B-it.litertlm \
  --system-file /opt/prompt/sys_prompt_manager.md \
  --enable-shell \
  --shell-cwd /opt/cat-agent \
  --shell-timeout 15 \
  --shell-max-output 16384
```

A suggested first user test: "Вызови shell и выполни pwd, сообщи результат."
The console must contain `native-tool CALL name=shell` and
`native-tool RESULT name=shell`; text from Gemma without those logs
is NOT proof of tool execution.

Return values: `ok`, `exit_code` (null on timeout/output overrun),
`output` (merged stdout/stderr) and `error` when unsuccessful. The process
is killed (including normally inherited children in its process group) on
timeout or output overflow. The defaults are 15 seconds and 16 KiB, with
respective maximum values 120 seconds and 64 KiB. Commands over 4096 chars,
empty commands and embedded NUL bytes are rejected.

**Security:** Enabling shell gives the model arbitrary Bash execution with
the effective permissions of the Python process. `--shell-cwd` only sets the
initial directory; it is **not a filesystem jail or access restriction**.
For production run under a dedicated low-privilege OS user/container and
consider an allow/approve policy for mutating commands. Never enable shell
in an exposed service without additional isolation and access control.

The `shell` callback is registered during BASE construction, but remains
**disabled during the one-time warm message**. It becomes available only
after a successful BASE warm and initial clone. On /new it remains available.

## One-time warm-up BASE (stock LiteRT-LM 0.18.x)

BASE is populated by one real service user turn on the base Conversation,
**before** any cloning. This works with the stock 0.18.0/0.18.1 wheel,
without a native rebuild:

1. Create BASE with the operator's system prompt and MQTT tools.
2. Send "warm" once to BASE, await the assistant reply.
3. Verify its native token_count increased and is greater than zero.
4. Clone the warmed BASE, using that clone as the working dialogue.
5. /new closes the working clone and clones BASE again; warm is not resent.

The service exchange ("warm" plus the assistant reply) is permanently part
of the BASE history. The operator may instruct Gemma in the system prompt to
reply exactly "OK" and not call any tools when receiving "warm". Python
does not require the reply to equal OK, but does require nonempty text.

Startup fails rather than silently using a cold BASE if native KV is absent,
or if clone KV does not match BASE. Both counts and the warm response are logged.
The legacy native_patch/ is retained for reference only; it is NOT required
or used. --warm-message overrides the default value "warm".

## WSL smoke test

Install MQTT client (Ubuntu): apt install mosquitto-clients
Use the Python interpreter with LiteRT-LM 0.18.0/0.18.1 installed and your
CPU-supported .litertlm model. The WSL host must reach the MQTT broker.

  cd /opt/cat-agent
  export PYTHONPATH=/opt/cat-agent/src
  /opt/litert/litert_env/bin/python3 -m litert_agent_tools \
      --model /storage/models/litert/gemma-4-E4B-it.litertlm \
      --system-file /opt/prompt/sys_prompt_manager.md

Override MQTT broker/port with CAT_AGENT_MQTT_HOST, CAT_AGENT_MQTT_PORT,
and LITERT_TOOLS_MQTT_WAIT (default 5 s).

The terminal keeps a dialogue until /new, which discards the user's conversation
while retaining the service warm/response exchange. /exit closes everything.

## Tests (do not require LiteRT or MQTT broker)

  cd /opt/cat-agent
  PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
      -m unittest discover -s tests -p 'test_litert_agent_tools*.py' -v

End-to-end native cloning, actual MQTT and KV prefill performance require a
real installed LiteRT-LM library, model and MQTT broker.

## Diagnosing real native tool calls

At the default INFO log level, native tool invocation is logged on both sides
of the subprocess execution:

    INFO:litert_agent_tools.mqtt_tools:native-tool CALL name=get_home_climate topic=zigbee2mqtt/temp_tua
    INFO:litert_agent_tools.mqtt_tools:native-tool RESULT name=get_home_climate result={"ok": true, "temperature_c": 24.66, "humidity_percent": 26.24}

This is an example only; real numbers must be obtained from the broker.
Clone events are also logged as native-conversation CLONE. If Gemma supplies
a number but no CALL log appears, it did NOT invoke this Python MQTT tool.

## Native KV counters

At INFO level, the sequence is:

    native-conversation BASE-WARM message='warm' response='OK' kv_before=0 kv_after=2345 elapsed=...
    native-conversation BASE-CLONE kv_tokens=2345 expected=2345
    native-conversation TURN kv_before=2345 kv_after=... last_prefill_tokens=... elapsed=...

The numbers above are examples, not actual measurements. Every /new creates
another clone starting with exactly the measured BASE count, without repeating
the warm message. last_prefill_tokens is only the last native prefill of a
turn: automatic Tool Calling can execute multiple prefill operations.

## Debugging native JSON parse failures

Use `--trace-native-json` only during debugging. This wraps the **working
Conversation's** LiteRT FFI object (not the native library), logging every
Python -> C send with its role (user or tool), byte count, SHA-256 prefix,
and a Python JSON validity check. If the C API rejects the input, it also
prints the first 4096 original JSON bytes. The log can contain private
command arguments and tool output; inspect before sharing.

If a shell callback was invoked with a malformed argument, the log
includes `native-tool DISPATCH name=shell arg_type=...` before validation.
The standard `native-tool CALL name=shell` appears only for valid commands.

Test command:
```sh
PYTHONPATH=/opt/cat-agent/src /opt/litert/litert_env/bin/python3 \
  -m litert_agent_tools \
  --model /storage/models/litert/gemma-4-E4B-it.litertlm \
  --system-file /opt/prompt/sys_prompt_manager.md \
  --enable-shell --shell-cwd /opt/cat-agent --trace-native-json
```

## Diagnosing a blank reply after a native tool call

With `--trace-native-json`, logs also show the raw LiteRT JSON response
structure after *each* model inference (initial tool call and follow-up).
`native-json RESPONSE` includes role, whether content is empty, tool call
count, and response fields. If the native final response is empty, the
`native-json EMPTY-FINAL` log includes the returned JSON bytes (up to
4096 bytes); the CLI then also writes `native-conversation EMPTY-REPLY`.
This distinguishes a native empty generation from a response-text extraction
problem. Instrumentation does not change the tool loop or send extra prompts.
