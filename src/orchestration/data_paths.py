from __future__ import annotations

from pathlib import Path


DATA_DIR_NAME = "data"


def data_root(runtime) -> Path:
    root = (runtime.root / DATA_DIR_NAME)
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"data directory does not exist: {root}") from exc
    if not resolved.is_dir():
        raise ValueError(f"data path is not a directory: {root}")
    return resolved


def resolve_data_path(runtime, value: str, *, must_exist: bool = False) -> Path:
    """Resolve /foo/bar as <workspace>/data/foo/bar.

    The leading slash is a logical DATA-root marker, not the Linux filesystem
    root. Parent traversal and symlink escape are rejected.
    """
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError("data path must start with /")
    logical = value[1:]
    if not logical:
        raise ValueError("data path must name a file")
    relative = Path(logical)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"invalid data path: {value}")

    root = data_root(runtime)
    raw = root / relative
    parent = raw.parent.resolve(strict=False)
    candidate = parent / raw.name
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"data path escapes data directory: {value}") from exc

    if candidate.is_symlink():
        raise ValueError(f"symlink is not permitted: {value}")

    if must_exist:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(f"data file does not exist: {value}") from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"data path escapes data directory: {value}") from exc
        if not resolved.is_file():
            raise ValueError(f"data path is not a regular file: {value}")
        return resolved

    return candidate
