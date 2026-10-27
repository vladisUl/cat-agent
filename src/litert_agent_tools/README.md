# Native tools experiment (LiteRT Conversation)

This package is deliberately separate from src/litert_agent, the OpenAI
client, MCP, /work#, and TASK SYSTEM. None of their code or prompts is modified.

## Scope

- Four native zero-argument Python tools registered in Conversation(tools=...):
  outdoor temperature/humidity, home temperature/humidity, aquarium temperature
  and freezer temperature. Topics are the existing entries in prompts/mqtt.md.
- Tools are read-only; each invocation runs mosquitto_sub for one MQTT JSON
  message. Errors are returned as JSON objects with ok=false.
- The temporary system prompt is supplied by the operator via --system-file.
  It contains no shell or /work# tool instructions.
- One BASE Conversation plus a disposable working clone. /new closes the old
  conversation and clones BASE, so its history and KV are independent.

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
      -m unittest discover -s tests -p 'test_litert_agent_tools.py' -v

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
