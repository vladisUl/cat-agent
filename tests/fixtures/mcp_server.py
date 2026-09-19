"""Real official-SDK server for stdio/HTTP integration tests and Radxa checks."""
import argparse
import asyncio
import os
from pathlib import Path
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

server = MCPServer("cat-agent-test", log_level="ERROR")


@server.tool()
def echo(text: str) -> str:
    """Return the supplied text unchanged."""
    return text


@server.tool()
def fail() -> str:
    """Return an intentional tool error."""
    raise ToolError("intentional test error")


@server.tool()
async def slow(marker: str) -> str:
    """Record one side effect, then sleep past the caller's timeout."""
    with Path(marker).open("a") as stream:
        stream.write("called\n")
    await asyncio.sleep(5)
    return "late"


@server.tool()
def process_id() -> int:
    """Report subprocess identity for shutdown checks."""
    return os.getpid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.port:
        server.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        server.run(transport="stdio")
