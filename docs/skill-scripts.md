# Internal skill scripts

A skill may be backed by a line-oriented scenario stored directly in the
workspace (normally `/opt/model`).

The scenario file name is `<skill-name>.sh`. It is interpreted as an internal
scenario only when `<skill-name>` is assigned to the current MANAGER/AGENT.
The file is not passed to bash.

Example:

```text
web_shot /browser/page.png
read_pic.sh /browser/page.png
```

The first token of every ordinary line names a real executable file in the
workspace root. Before launch runtime verifies that the file exists, is a
regular file, is not a symlink, and has execute permission. The file extension
has no special meaning: binaries, shell scripts and executable Python scripts
with a shebang are handled identically.

Arguments beginning with `/` are logical DATA paths. Runtime maps them under
`<workspace>/data` before starting the external program:

```text
/browser/page.png -> /opt/model/data/browser/page.png
```

The external process runs with cwd set to the workspace root.

`read_pic.sh` is an internal opcode. It never reaches Linux. It attaches the
PNG/JPEG named by its logical DATA path to the current model context using the
existing multimodal adapter.

Blank lines and full-line comments beginning with `#` are ignored. Current
version executes lines sequentially. Control flow is intentionally not defined
yet; the parser is separate from bash so conditions can be added later without
changing the external-command or DATA-path contracts.

DATA paths cannot escape `<workspace>/data` through `..` or symlinks.
