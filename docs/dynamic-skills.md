# Dynamic skills

Dynamic skills are startup-discovered capabilities stored in `skills/*.txt`.
They are frozen for the lifetime of a CORE, like the MCP catalog.

Example `skills/prognoz.txt`:

```text
code: /work#prognoz.sh
description: прогноз на 3 дня в городе
manager: true
```

The file name defines the skill name: `prognoz.txt` -> `prognoz`.
For now `code` must be exactly `/work#<skill-name>.sh`.

All dynamic skills are assignable to AGENT tasks. When a dynamic skill is
delegated, its generated instructions including `code` are included in that
agent's system context.

`manager: true` also gives MANAGER the direct invocation code.
`manager: false` exposes only the skill name and description to MANAGER so it
can delegate the capability without receiving its direct command.

At execution time an assigned dynamic command such as `/work#prognoz.sh`
causes runtime to read `<workspace>/prognoz.sh` (normally
`/opt/model/prognoz.sh`) as the internal line-oriented skill scenario.
A missing scenario is an error.

Scenario lines are dispatched in order:

- `read_pic.sh` is an internal image opcode.
- `mqtt_sub.sh` and `mqtt_pub.sh` are internal MQTT pseudo-commands.
- Any other first token must name an existing executable regular file in the
  workspace root.

External command file arguments are resolved under `<workspace>/data`.
The DATA directory is intended to be tmpfs and may contain subdirectories.

Adding or changing a file under `skills/` requires a CORE restart so the
frozen startup catalog and model base remain stable.
