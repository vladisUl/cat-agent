# MQTT Native Tools

`config/mqtt.json` declares `port`, `timeout`, and named `servers` (e.g. `old_server`, `new_server`). Servers can be a hostname string or `{"host":"example","port":1884}` if a broker uses a different port.

`config/mqtt_tools.json` contains `{"tools":[...]}`. Each entry defines `name`, `description`, `server`, `topic`, and `fields` (source JSON field → result field). Optional `validation` supports `number`, `boolean`, `string`, and numeric `min`/`max`. Change or add entries without changing Python.

`create_mqtt_tools()` loads both documents at startup and creates one ordinary no-argument Python callable per entry, with JSON-derived `__name__`, `__doc__` and a closure over MQTT parameters. The result is the native LiteRT `tools: list`; no Python files and no classes derived from `litert_lm.Tool` are generated. The model never supplies topics or broker addresses.

The registered names are shared with MANAGER, AGENT, and TASK worker permissions. Restart Native CORE after changing JSON.


## Native external MQTT events (legacy cycle -1)

The shared **TASK SYSTEM** process owns the MQTT subscription, the durable
event bindings and the RUN queue. Native LiteRT CORE does **not** create another
event listener. `config/mqtt_tools.json` is the native tool catalog; the
OpenAI `/opt/prompt/prompts/mqtt.md` catalog is unchanged.

Two named boolean sensors are provided:

| Tool name | MQTT topic | JSON field | Event values |
| --- | --- | --- | --- |
| `get_corridor_motion` | `zigbee2mqtt/dvigen_verh` | `occupancy` | `true` movement, `false` none |
| `get_kitchen_leak` | `zigbee2mqtt/kuhnya_leak` | `water_leak` | `true` leak, `false` dry |

Native MANAGER creates a saved external QUERY through the **same**
`delegate_task` tool used for one-shot and periodic delegation:

```text
delegate_task(agent_id="agent_1", task="Сообщить о движении в коридоре",
              period_seconds=-1, mqtt_tool="get_corridor_motion")
```

The compatibility sentinel is `period_seconds=-1` (external event),
`0` remains one-shot, and positive periods remain timer schedules. The
`mqtt_tool` selector is mandatory only for `-1`; it selects the topic and
source field from the validated native catalog, never from arbitrary model
input. Only a one-field `boolean` MQTT tool can be selected. The agent must
have permission for that tool, and the configured sensor broker must match
the MQTT broker monitored by TASK SYSTEM (`CAT_AGENT_MQTT_HOST` /
`CAT_AGENT_MQTT_PORT`, currently 192.168.0.21:1883 by default).
Unmatched brokers fail creation instead of silently never triggering.

TASK SYSTEM saves the task with `timer_period_seconds=null` (not an invalid
negative timer) and exposes it via native query management as period `-1`.
The same `list_periodic_queries`, `start_periodic_query`,
`stop_periodic_query` and `delete_periodic_query` controls manage either
kind of saved native QUERY. Native queries are pinned to the LiteRT executor;
the existing OpenAI tasks and their configuration are not modified.

The MQTT monitor emits field **changes** in either direction. For the first
received payload after startup, `config/mqtt_event_active.json` permits only
the configured active state (`true` for both of these sensors) to trigger.
The worker schedules the native agent with hardware-event priority and
includes the exact queued `topic`, `field` and `value` in its instruction.
It does not query the sensor again: the current state could have changed before
the agent was scheduled. QUERY replies follow existing TASK notifications.

After deploying, restart `./start_litert_core.sh` so the native model
receives the two new tool declarations and the extended `delegate_task`
signature. TASK SYSTEM itself is unchanged: a running instance already listens
to MQTT and picks up new bindings from its in-memory registry immediately.
Start `./start_system_core.sh` only if TASK SYSTEM is not already running.
Existing periodic tasks remain valid.
