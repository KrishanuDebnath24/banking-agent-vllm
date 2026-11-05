"""python -m api   ->  http://127.0.0.1:8000  (API_HOST / PORT to change)"""

import asyncio
import os

import uvicorn

if __name__ == "__main__":
    from api.app import app  # loads .env
    import storage

    host, port = os.environ.get("API_HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))
    print(f"Banking assistant: http://{host}:{port}   (the bank servers must be running: python -m mcp_servers.run)")
    print(f"Records: {storage.describe()}")
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level=os.environ.get("LOG_LEVEL", "warning")))
    factory = storage.loop_factory()  # Windows + Postgres needs the selector event loop
    if factory:
        asyncio.run(server.serve(), loop_factory=factory)
    else:
        server.run()
