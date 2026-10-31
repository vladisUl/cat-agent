# MQTT Native Tools

`config/mqtt.json` declares `port`, `timeout`, and named `servers` (e.g. `old_server`, `new_server`). Servers can be a hostname string or `{"host":"example","port":1884}` if a broker uses a different port.

`config/mqtt_tools.json` contains `{"tools":[...]}`. Each entry defines `name`, `description`, `server`, `topic`, and `fields` (source JSON field → result field). Optional `validation` supports `number`, `boolean`, `string`, and numeric `min`/`max`. Change or add entries without changing Python.

`create_mqtt_tools()` loads both documents at startup and creates one ordinary no-argument Python callable per entry, with JSON-derived `__name__`, `__doc__` and a closure over MQTT parameters. The result is the native LiteRT `tools: list`; no Python files and no classes derived from `litert_lm.Tool` are generated. The model never supplies topics or broker addresses.

The registered names are shared with MANAGER, AGENT, and TASK worker permissions. Restart Native CORE after changing JSON.
