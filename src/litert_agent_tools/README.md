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

## IMPORTANT: BASE prefill limitation

LiteRT-LM 0.18.0 and 0.18.1 have native C Conversation.clone but no Python
wrapper. clone.py exposes this API using its 0.18-specific signature; versions
with a Python .clone() use that method instead.

In LiteRT-LM 0.18.x, Python create_conversation() does not expose a
prefill-preface-on-init switch. BASE cloning works but the system prompt has
NOT been prefilled yet when BASE is cloned. Accordingly this experiment does
NOT claim that its independent turns already avoid reprocessing the system
prompt. Proving a clean prefilled BASE with native Conversation remains a
separate engineering milestone. We do not fabricate a dummy assistant exchange
or silently modify the old Session checkpoint implementation.

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
