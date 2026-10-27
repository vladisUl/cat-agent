"""Start the isolated CORE skeleton: python -m agent_core_tools."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal
import sys

from .server import CoreServer, DEFAULT_SOCKET


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native-tools CORE skeleton (no model yet)")
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    server = CoreServer(args.socket)

    def terminate(_signal, _frame) -> None:
        server.close()

    signal.signal(signal.SIGTERM, terminate)
    try:
        server.start()
        server.serve_forever()
    except OSError as exc:
        print(f"TOOLS CORE startup/service error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
