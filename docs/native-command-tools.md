# Native LiteRT command tool factory

The command-tool factory is independent from MQTT and the legacy OpenAI CORE.
Edit `config/command_tools.json` to register executables; restart Native CORE.

Executable directory: `litert.command_dir` in `cat-agent.yaml`. Files must be
executable and have a valid shebang if they are scripts. A tool name may differ
from the executable basename.

Example (documentation only; no example tools are automatically installed):

```json
{
  "tools": [
    {
      "name": "get_cat_water_level",
      "description": "Получить уровень воды у кота в указанной комнате.",
      "command": "get_cat_water_level",
      "parameters": {
        "room": {
          "type": "string",
          "enum": ["kitchen", "bedroom"],
          "description": "Комната для измерения."
        }
      }
    },
    {
      "name": "get_cat_food_level",
      "description": "Получить остаток корма в кормушке.",
      "command": "get_cat_food_level"
    }
  ]
}
```

For `get_cat_water_level(room="kitchen")` the factory executes
`<litert.command_dir>/get_cat_water_level kitchen` with `shell=False`.
For the no-argument tool it starts `get_cat_food_level` with no arguments.

Tools are registered as ordinary typed Python callables and shared by MANAGER
and all native AGENTs, just like MQTT tools. `name`, `description`
and `parameters` are all the model needs; executable paths are internal.

All parameters are required, passed as argv values in JSON declaration order.
Supported types: `string`, `integer`, `number`, `boolean`.
`enum` optionally restricts values. Programs write results to stdout,
diagnostics/errors to stderr. JSON stdout becomes a structured `result`;
other stdout stays a string (e.g. `70%`). On success the tool returns
`{"ok": true, "result": "70%"}`; on failure it returns `ok=false`
with an `error`, and exit status when available.

The factory enforces the same timeout and output-size limits configured for
Native shell (`--shell-timeout`, `--shell-max-output`), without using
a shell interpreter. The warm-up gate prevents executable calls before
the Native manager finishes BASE creation.

## Initial date/time tool

`get_date_time()` is the first configured tool. Its executable is included
as `native_commands/get_date_time`, and delegates to `/usr/bin/date`.

Install the executable once in the directory configured as `litert.command_dir`
(currently `/opt/model/bin`), before restarting Native CORE:

```bash
install -D -m 755 /opt/cat-agent/native_commands/get_date_time /opt/model/bin/get_date_time
```

The tool returns local server time with its numeric time zone offset as text
(e.g. `2026-10-31 19:30:00 +0500`). Further executables can be added by
copying them into the same directory and editing the JSON definitions.
