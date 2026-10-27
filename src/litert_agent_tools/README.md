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

## Genuine BASE prefill (opt-in native source extension)

LiteRT-LM 0.18.0 and 0.18.1 have native C Conversation.clone but no Python
wrapper. clone.py exposes the 0.18-specific C signature. Newer releases
implementing Conversation.clone() in Python use that method.

The STOCK 0.18.x C/Python API does not expose C++ SetPrefillPrefaceOnInit.
Therefore a stock wheel can only clone an unprefilled BASE: no cache savings
are claimed for the system prompt. At startup, BASE KV tokens are read from
the native library; they are not inferred from prompt text.

The optional minimal source patch under native_patch/ adds that missing C/Python
binding WITHOUT changing the C++ prefill implementation and WITHOUT a synthetic
dialogue. Rebuilding LiteRT-LM with the patch enables real BASE prefill; the
standalone tool automatically detects the new Python argument and requests it.
It then verifies that each clone has inherited the same positive KV length.

See [native_patch/README.md](native_patch/README.md). Native rebuilding is
required; Python code alone cannot enable the missing C ABI option.

Run with --require-prefilled-base to fail explicitly if the installed library
is not patched.

## WSL smoke test

Install MQTT client (Ubuntu): apt install mosquitto-clients
Use the Python interpreter with LiteRT-LM 0.18.0/0.18.1 installed and your
CPU-supported .litertlm model. The WSL host must reach the MQTT broker.

  cd /opt/cat-agent
  export PYTHONPATH=/opt/cat-agent/src
  /opt/litert-lm-venv/bin/python3 -m litert_agent_tools \
      --model /path/to/model.litertlm \
      --system-file /path/to/temporary_system_prompt.txt

Override MQTT broker/port with CAT_AGENT_MQTT_HOST, CAT_AGENT_MQTT_PORT,
and LITERT_TOOLS_MQTT_WAIT (default 5 s).

The terminal keeps a dialogue until /new, which discards question, answer and
tool-result history by closing the working clone. /exit closes everything.

## Tests (do not require LiteRT or MQTT broker)

  cd /opt/cat-agent
  PYTHONPATH=src python3 -m unittest discover -s tests \
      -p 'test_litert_agent_tools.py' -v

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

At INFO level the console prints:

    native-conversation BASE prefill_supported=False kv_tokens=0 elapsed=...
    native-conversation BASE-CLONE kv_tokens=0 expected=0
    native-conversation TURN kv_before=0 kv_after=... last_prefill_tokens=... elapsed=...

The zero counts above are an illustrative STOCK 0.18 library case. With the
patched native library, BASE and every fresh BASE-CLONE should be >0 and equal.
The last_prefill_tokens metric covers only the last prefill operation of a
turn; automatic tool calling may trigger multiple prefill operations.
