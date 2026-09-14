"""Runtime implementation of read_pic.sh; image bytes stay in dialogue history."""
from __future__ import annotations

import base64
import os
import shlex


def read_picture(command, runtime, client):
    """Return message content for this tool, or None for an ordinary command."""
    if command.strip().split(maxsplit=1)[:1] != ["read_pic.sh"]:
        return None
    try:
        argv = shlex.split(command)
        if len(argv) != 2:
            raise ValueError("usage: read_pic.sh NAME.png|NAME.jpg")
        if getattr(client, "supports_images", False) is not True:
            raise ValueError("read_pic is supported only by LiteRT-LM and OpenAI backends")
        if "read_pic" not in runtime.skill_names:
            raise ValueError("read_pic is not assigned to this agent")
        enabled = {name.strip() for name in os.getenv("CAT_AGENT_ENABLED_SKILLS", "shell,mqtt,read_pic").split(",")}
        if "read_pic" not in enabled:
            raise ValueError("read_pic disabled by runtime policy")
        path = (runtime.cwd / argv[1]).resolve(strict=True)
        if not path.is_relative_to(runtime.root) or not path.is_file():
            raise ValueError("image must be a regular file inside the workspace")
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            raise ValueError("only PNG and JPEG images are supported")
        limit = int(os.getenv("CAT_AGENT_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
        if limit <= 0:
            raise ValueError("CAT_AGENT_MAX_IMAGE_BYTES must be positive")
        with path.open("rb") as source:
            data = source.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"image exceeds {limit} bytes")
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            mime = "image/png"
        elif data.startswith(b"\xff\xd8\xff"):
            mime = "image/jpeg"
        else:
            raise ValueError("file does not contain a PNG or JPEG image")
        encoded = base64.b64encode(data).decode("ascii")
        return [
            {"type": "text", "text": f"read_pic.sh: {path.name}\nImage attached. Continue the current task using this image."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]
    except (OSError, ValueError, RuntimeError) as exc:
        return f"SYSTEM_ERROR\nread_pic.sh: {exc}"
