# Dynamic skills

Local tool definitions now live in `tools/*.md`; see the canonical format in
[README](../README.md#tools). The former `skills/*.txt` loader has been removed;
that directory is not a runtime catalog source.

All local tools can be assigned to AGENT tasks. `manager: true` includes the
verbatim tool block in MANAGER's prompt. The catalog is frozen at CORE startup.

At execution time an assigned dynamic command such as `/work#prognoz.sh`
causes runtime to read `<workspace>/prognoz.sh` (normally
`/opt/model/prognoz.sh`) as the internal line-oriented skill scenario.
A missing scenario is an error.

Text appended after the skill command is the scenario input. Runtime keeps this
tail opaque and passes it to the first scenario step on stdin. For example all
of these are valid skill invocations when the skill itself understands the
corresponding format:

```text
/work#my_skill.sh 3
/work#my_skill.sh -f file.txt
/work#my_skill.sh -j test.json
/work#my_skill.sh "привет" -f output.txt
```

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

Scenario text flows from step to step: stdout of an external step becomes stdin
of the next step. Internal tools participate in the same chain through their
existing runtime adapters. A step may pass the incoming text through, replace
it, or emit something new such as the name of a generated file.

The final textual pipeline output is returned as the skill result. Empty final
output means successful silent completion; runtime does not invent
`SYSTEM_OK`. Authors should print at least `OK` from the final text-producing
step when an explicit acknowledgement is desired.

`read_pic.sh` is the current non-text result type. Inside a scenario it may use
an explicit path (`read_pic.sh file.png`) or, when written bare, consume the
previous step's output as its path. Because it produces a multimodal result it
must be the final scenario step.

Adding or changing a tool definition under `tools/` requires a CORE restart so the
frozen startup catalog and model base remain stable.
