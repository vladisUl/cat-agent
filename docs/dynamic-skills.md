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

A skill may exist only for side effects: creating or changing files, sending
commands to peripherals, publishing MQTT data, and similar work. Those actions
are logged but do not themselves create a user-visible result.

The textual result of a scenario is the stripped stdout of its **last external
command**. It is deliberately not the last non-empty stdout: if the last
external command succeeds with empty stdout, the skill completes silently.
If there are no external commands producing a textual result, it also completes
silently. Authors should print at least `OK` from the final external command
when an explicit acknowledgement is desired.

`read_pic.sh` is the current special result type: when used successfully, the
scenario returns the attached image instead of a textual stdout result.

Adding or changing a file under `skills/` requires a CORE restart so the
frozen startup catalog and model base remain stable.
