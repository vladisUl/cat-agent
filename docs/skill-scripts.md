# Internal skill scripts

A dynamic skill may be backed by a line-oriented scenario stored directly in
the workspace (normally `/opt/model`).

The scenario file name is `<skill-name>.sh`. It is interpreted as an internal
scenario only when `<skill-name>` is assigned to the current MANAGER/AGENT.
The file is not passed to Bash.

## Skill input

Everything after the scenario name in the `/work#` command is treated as
opaque input text for the scenario.

Examples:

```text
/work#my_skill.sh 3
/work#my_skill.sh -f file.txt
/work#my_skill.sh -j test.json
/work#my_skill.sh "привет" -f output.txt
```

cat-agent does not interpret that tail as filenames, options, JSON or any other
application-specific format. The text is passed to the first scenario step on
stdin. The first executable decides how to parse it.

This keeps the scenario interface independent of the command-line syntax used
by any particular helper program.

## Pipeline

Scenario lines execute sequentially as a text pipeline:

```text
skill input
    |
    v
step 1 stdin
    |
    +-- stdout
            |
            v
        step 2 stdin
            |
            +-- stdout
                    |
                    v
                 ...
```

For external executables, stdout is passed unchanged to stdin of the next step.
The final textual result returned by the whole skill is the stripped output of
the final text-producing step.

A step may therefore pass the incoming data through, transform it, or replace
it completely. For example a program may receive `7day`, create an image and
print only the generated filename. A following `read_pic.sh` can consume that
filename.

If the final text output is empty, the skill completes successfully and
silently. Runtime does not invent `SYSTEM_OK`. Printing `OK` from the final
step is recommended when an explicit acknowledgement is desired.

## External steps

The first token of every ordinary scenario line names a real executable file in
the workspace root. Before launch runtime verifies that the file exists, is a
regular file, is not a symlink, and has execute permission. The file extension
has no special meaning: binaries, shell scripts and executable Python scripts
with a shebang are handled identically.

Arguments written directly in the scenario line remain static arguments of that
step. Existing file-argument handling is unchanged: non-option arguments are
logical DATA paths and are resolved under `<workspace>/data` before launch.

Example:

```text
web_shot /browser/page.png
read_pic.sh /browser/page.png
```

The external process runs with cwd set to the workspace root.

## Internal tools in a pipeline

The scenario runner reuses the same internal tool implementations used by
independent `/work#` calls. It does not maintain second copies of their logic.

`read_pic.sh` supports both forms:

```text
read_pic.sh picture.png
```

uses the explicit argument, while:

```text
generator
read_pic.sh
```

uses the previous step's stdout as the image path.

An explicit argument always wins over pipeline input. A successful
`read_pic.sh` returns a multimodal result, not text, so it must be the final
scenario step.

The MQTT pseudo-commands follow the same rule: explicit arguments are used when
present; a bare `mqtt_sub.sh` or `mqtt_pub.sh` may consume the previous text
output as its argument string. Their existing runtime implementation and policy
checks remain shared with independent tool calls.

## DATA paths

Logical DATA paths may be written with or without a leading `/`. Both forms
map under `<workspace>/data`:

```text
browser/page.png  -> /opt/model/data/browser/page.png
/browser/page.png -> /opt/model/data/browser/page.png
```

Subdirectories are supported. DATA paths cannot escape `<workspace>/data`
through `..` or symlinks.

Blank lines and full-line comments beginning with `#` are ignored. Current
version executes lines sequentially. Control flow is intentionally not defined
yet.

## Fragment processing and cancellation

`read_partly.sh` and the `cyclic_process` scenario primitive yield to CORE after
each child model response and BASE reset. The manager and agent both use this
cooperative path. A higher-priority request can run before the next fragment;
TASK validity is checked again at the boundary. A model call already in progress
finishes before cancellation is applied. This does not introduce parallel model
inference or automatic replay of a cancelled TASK.

Closing a suspended operation closes its child model context. `cyclic_process`
removes its temporary output; an existing final output is replaced only after
successful completion of all fragments. The child is also closed if temporary
file creation fails.

File loading, splitting, and output formats are unchanged: `read_partly` reads
and splits the whole source in RAM and returns the joined fragment responses;
`cyclic_process` returns the output filename.

CORE status/snapshot includes `fragment_runs` for active and suspended requests:
`request_id`, `label`, `tool`, `index`, `total`, `source`, `phase` and
`cancel_pending`. Phases are `running` and `between_fragments`.
`cancel_pending` indicates a locally requested cancellation or CORE shutdown;
remote TASK STOP/DELETE is detected at the next boundary, not during inference.

Read this diagnostic without acquiring the human session (choose either socket):

```bash
/opt/litert-lm-venv/bin/python3 - /run/cat-agent/litert.sock <<'PY'
import json
import socket
import sys

with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.settimeout(5)
    client.connect(sys.argv[1])
    client.sendall(b'{"type":"snapshot"}\n')
    with client.makefile('r', encoding='utf-8') as stream:
        for line in stream:
            message = json.loads(line)
            if message.get('type') == 'snapshot':
                status = message['status']
                print(json.dumps({key: status.get(key) for key in
                                  ('manager_kv', 'fragment_runs')},
                                 ensure_ascii=False, indent=2))
                break
PY
```

After a failed KV1 reset, the next new interactive request rebuilds its BASE.
Successful recovery logs `CORE manager KV1 recovered`. If rebuilding fails,
KV1 remains unavailable, KV2 may serve the request, and recovery is retried on
a later new request. Previously executed turns are never replayed.
