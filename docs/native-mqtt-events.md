# Native LiteRT MQTT events

Native `delegate_task` supports the existing schedule convention:

* `period_seconds=0`: one-shot delegation;
* `period_seconds>0`: periodic TASK (native range remains 10..604800 seconds);
* **`period_seconds=-1`: persistent external MQTT event TASK**.

The native catalogue is `config/mqtt_tools.json`. Boolean fields declared in
`validation` can be selected as event sources. The manager receives the available
names and field descriptions automatically; `/opt/prompt/prompts/mqtt.md` remains
the OpenAI catalogue and is not modified or required for native registration.

Included sources:

| Tool | Topic | Field |
| --- | --- | --- |
| `get_corridor_motion` | `zigbee2mqtt/dvigen_verh` | `occupancy` |
| `get_kitchen_leak` | `zigbee2mqtt/kuhnya_leak` | `water_leak` |

Example native call:

```python
delegate_task(
    agent_id="agent_1",
    task="Если на кухне обнаружена протечка, сообщи мне. Если сухо, молчи.",
    period_seconds=-1,
    event_tool="get_kitchen_leak",
    event_field="water_leak",
)
```

`event_field` may be omitted when the selected tool has exactly one boolean
field. It must be the original MQTT field name, not a renamed output key.
The selected agent must have that MQTT tool registered. Unknown fields/tools,
ambiguous fields, and disabled tool grants are rejected before task creation.

The common TASK SYSTEM creates the TASK and event binding in one transaction.
It owns the only event subscription and dispatches native runs with
`executor=litert`, the selected `agent_id`, and `hardware` priority. The model
executor remains serialized: an event does not interrupt native decoding already
in progress. This path is independent of the optional MQTT **read cache**.

The existing monitor behavior is preserved: changes to both `true` and `false`
produce events; repeated identical values do not. On the first observed payload,
an event is emitted only when its value matches `config/mqtt_event_active.json`
(currently `true` for both sources). A retained active value can therefore
trigger on TASK SYSTEM startup. The native worker reads the same active-state
configuration at startup: the inactive transition is **rearm only**. It finishes
that RUN successfully without calling the model or publishing a notification.
For the supplied motion and leak sensors, `true` invokes the agent and `false`
is silent, regardless of task wording or model behavior. The shared monitor and
OpenAI event processing are unchanged. Signals absent from the active-state
configuration retain their previous processing behavior.

The saved instruction is followed by a `[SYSTEM_MQTT_EVENT]` block containing
topic, field, and value **at event time**. An agent must not substitute a later
sensor reading for this value. Nonempty successful agent replies use the existing
QUERY notifications inbox; a silent completion does not send a notification.

During that AGENT execution, a native MQTT read tool targeting the same broker,
topic, and single event field returns the event-time value directly, with
`source: "mqtt_event"`. It does not subscribe again or read the optional cache.
This also handles a model that calls the sensor tool despite the event block.
The scoped value is cleared after success or failure. Other sensors and ordinary
human/timer turns keep their normal read behavior. A multi-field tool is not
filled with invented values for fields absent from the event.

Use the existing `list_periodic_queries`, `stop_periodic_query`,
`start_periodic_query`, and `delete_periodic_query` tools for both timer and event
tasks. Event tasks show `period_seconds=-1`; they have no timer in storage.
Pause disables event admission, resume waits for subsequent events, and delete
removes the binding. TASKs and bindings survive TASK SYSTEM restarts.

The shared monitor currently subscribes to one broker, configured by
`CAT_AGENT_MQTT_HOST`/`CAT_AGENT_MQTT_PORT` (defaults 192.168.0.21:1883).
Both supplied sources use `old_server` in `config/mqtt.json`. Native event
registration rejects a source on a different broker. If overriding these
environment variables, use the same values for TASK SYSTEM and Native CORE.

## Update and verification

Pull the changes and restart Native CORE so its BASE contains the new catalogue
and tool schema. No TASK SYSTEM implementation change or second monitor is
required. Existing OpenAI event tasks, their `-1` convention and bindings are
unchanged.

Ask the manager to watch motion or leaks, then confirm that its saved-task list
shows `period_seconds=-1`. Trigger the actual sensor and inspect the QUERY inbox.
Tests exercise the real SQLite TASK engine and native worker with injected MQTT
messages; live broker and model testing remains a target-machine check.

## Offline notifications

Native TASK QUERY results go to the connected native interface. If no native
interface is connected (or its socket write fails), CORE first saves the text
in the shared notification outbox, then acknowledges the TASK SYSTEM inbox.
The shared dispatcher uses an active interface on another CORE when available;
otherwise it sends through the same Firebase transport as OpenAI CORE.

The existing `CAT_AGENT_OUTBOX` path (default
`/var/lib/cat-agent/outbox.sqlite3`) and `CAT_AGENT_FIREBASE_*` settings are reused.
All CORE processes must use the same outbox file for shared routing. Default
Firebase files remain `/opt/firebase/zigbee.json` and `/opt/firebase/tokens.txt`;
`firebase-admin` must be installed in the Python environment running native CORE.

A failed send stays queued with retry backoff. The worker periodically recovers
unacknowledged TASK inbox results while no native interface is connected,
including after restart or a disconnect before display acknowledgement.
A stable notification ID derived from the RUN prevents repeated handoffs from
creating duplicate queue entries. Delivery remains at-least-once: a lost provider
response can cause a repeated push. Firebase acceptance confirms provider receipt,
not display on the phone. Inactive MQTT edges still produce no notification.
