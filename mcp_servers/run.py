"""Start the MCP servers.

    python -m mcp_servers.run                 # all in one process: 3 bank MCP servers (8101-8103),
                                              # the security code service (8104), charts (8105)
    python -m mcp_servers.run --only service  # just one, e.g. to run them in separate terminals

Leave it running while you chat. Servers listen on 127.0.0.1 only (MCP_HOST changes that; Docker
sets it so the web chat container can reach them).
Set OBP_FAKE=1 to use the built-in fake bank instead of the OBP sandbox.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import os

import uvicorn

from mcp_servers.common import PORTS

MODULES = {
    "accounts": "mcp_servers.accounts.server",
    "transactions": "mcp_servers.transactions.server",
    "service": "mcp_servers.services.server",
    "verify": "mcp_servers.verification",   # security codes: plain HTTP for the chat app, not MCP
    "charts": "mcp_servers.charts.server",  # draws figures the transactions server saved; no bank access
}


async def serve(names: list[str]) -> None:
    host = os.environ.get("MCP_HOST", "127.0.0.1")
    servers = []
    for name in names:
        module = importlib.import_module(MODULES[name])
        if name == "verify":
            app, path = module.app, "/verify"
        else:
            app, path = module.mcp.streamable_http_app(), "/mcp"
        config = uvicorn.Config(app, host=host, port=PORTS[name], log_level="warning")
        servers.append(uvicorn.Server(config))
        print(f"  {name:<13} http://{host}:{PORTS[name]}{path}")
    print("Bank: " + ("FAKE (OBP_FAKE=1)" if os.environ.get("OBP_FAKE") == "1" else "OBP sandbox")
          + ". Press Ctrl+C to stop.")
    if "verify" in names:
        from mcp_servers.verification import summary
        print(summary())
    import storage
    print(f"Records: {storage.describe()}")
    await asyncio.gather(*(s.serve() for s in servers))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(PORTS), action="append", help="start only these servers")
    args = ap.parse_args()
    print("Starting MCP servers:")
    try:
        asyncio.run(serve(args.only or list(PORTS)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
