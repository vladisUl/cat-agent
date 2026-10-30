"""Attach local DATA images as native LiteRT Conversation message content.

No Tool, shell or intermediate model call. Preserve the complete user text.
The file is resolved by the same validator as legacy read_pic.sh.
"""
from __future__ import annotations

from pathlib import Path
import re
import shlex
from types import SimpleNamespace

from orchestration.image_tool import validated_picture_path


# Single filename reference in an ordinary question (including DATA subfolders).
# Quoted paths containing spaces are supported through explicit read_pic.sh syntax.
PICTURE_REF = re.compile(
    r"(?<![\\w./-])/?(?:[\\w.-]+/)*[\\w.-]+\\.(?:png|jpe?g)(?![\\w-])",
    re.IGNORECASE,
)


def picture_path_in_text(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            argv = shlex.split(line)
        except ValueError:
            argv = []
        if len(argv) == 2 and argv[0] in ("read_pic.sh", "/work#read_pic.sh"):
            return argv[1]
    matches = list(dict.fromkeys(PICTURE_REF.findall(text)))
    if len(matches) > 1:
        raise ValueError("Specify one PNG/JPEG file per image request")
    return matches[0] if matches else None


def native_picture_message(text: str, workspace: Path):
    """Build one user message with original question and an ImageFile path.

    Return the original string unchanged for ordinary text turns.
    """
    logical_path = picture_path_in_text(text)
    if logical_path is None:
        return text
    path = validated_picture_path(SimpleNamespace(root=Path(workspace)), logical_path)
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image", "path": str(path)},
        ],
    }
