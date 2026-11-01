# Native tool calling limits

The native CORE still uses stock LiteRT-LM automatic tool calling, warmed
BASEs and independent clones. `tool_event_handler` bounds dispatch and tool
results; no SDK installation or native ABI is patched. Compatible hook placement
was checked against LiteRT-LM 0.18.0 and 0.19.0.

## Configuration

`litert.native` in `cat-agent.yaml` controls the native CORE independently of
the legacy `agent` settings:

```yaml
native:
  max_context_tokens: 4096
  max_output_tokens: 512
  max_tool_calls: 12
  tool_timeout: 180
  max_tool_result_bytes: 2048
  temperature: 0.0
  top_p: 1.0
```

The launcher forwards these values to the CLI and the runtime logs the applied
settings. `max_tool_calls` includes unsuccessful dispatches and unknown tool
names. A manager turn shares the budget across its delegated agents and result
reports. Independent background agent turns receive a fresh budget.

The timeout and client cancellation are checked at tool boundaries. They do not
interrupt native decoding or kill an already running callback. Shell commands
retain their separate subprocess timeout. A limit aborts the automatic loop;
the existing CORE/AGENT recovery discards the working context, never replays
the operation, and keeps BASE. An operation completed before interruption may
already have changed external state.

## Result budget

Each tool response is limited by both `max_tool_result_bytes` and remaining
resident context after reserving `max_output_tokens` and formatting headroom.
Responses in a single batch share the remaining budget. The estimator counts
escaped JSON bytes conservatively; it is **not an exact model tokenizer** and
does not guarantee that arbitrary user text or image input fits the context.

Oversized responses are saved as UTF-8 JSON under
`<workspace>/DATA/tool-results/<result_id>.json`. The model receives
`truncated: true`, status fields, a preview and a `result_id`.
`read_tool_result(result_id, offset, length)` reads the saved result without
re-executing the original command. Offset and length are Unicode character
counts. Read only relevant pages; reading an entire large result into the same
dialogue still consumes context. Files are retained until the operator removes
them; they can contain command output and should be included in normal workspace
storage housekeeping.

This preserves the complete result captured by the tool, not output beyond the
existing subprocess output cap. A subprocess exceeding its output cap still
fails with `output_limit_exceeded`.

## Selective agent tools

Existing agent definitions continue to inherit MQTT and command tools.
To register only explicitly listed tools for an agent, add
`"inherit_shared_tools": false` alongside `id`, `prompt`, and `tools`.
The saved-result reader remains available. TASK worker permissions use the same
selection as Conversation registration.

## Optional persistent MQTT subscriptions

Install the optional dependency in the native virtual environment:

```bash
/opt/litert-lm-venv/bin/python3 -m pip install '/opt/cat-agent[mqtt-cache]'
```

Add `"cache_max_age_seconds": 300` to `config/mqtt.json` to enable one persistent
client per broker and reuse its subscriptions across tools/agents. Choose the
age limit for your sensors' publication intervals; 300 is an example, not an
assumption about their configuration. Zero or absence preserves `mosquitto_sub`
one-shot reads and requires no extra dependency.

Disconnected clients discard cached values. Expired or missing values return
`ok: false`; they are not silently served or replaced with one-shot reads.
`received_age_seconds` measures time since local receipt, **not sensor sampling
time**. `retained: true` identifies a broker-retained message whose original
publication age is unknown. Network loss is detected by the MQTT client and
keepalive, not instantaneously. The clients stop when CORE closes.

Performance gains require measurement on the target machine; the cache is
implemented but disabled by default.
