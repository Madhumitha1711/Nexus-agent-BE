import threading
import time
import socket

from src.api.mcptools import mcp
from uvicorn import run as uvicorn_run

# Importing app triggers server setup and lifespan-registration
from src.api import server as server_module


def start_mcp_server():
    # Start MCP server using streamable HTTP transport on default port (8001)
    # If you need a custom port, you can pass transport options accordingly.
    mcp.run(transport="streamable-http")


def wait_for_port(host: str, port: int, timeout_seconds: int = 30, interval: float = 0.2) -> bool:
    """Wait until a TCP port is accepting connections."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(interval)
    return False


def main():
    # Start MCP in a daemon thread so FastAPI can run in the main thread
    t = threading.Thread(target=start_mcp_server, name="mcp-server-thread", daemon=True)
    t.start()

    # Block until MCP is ready
    ready = wait_for_port("127.0.0.1", 8000, timeout_seconds=45, interval=0.25)
    if not ready:
        # Log and proceed anyway to avoid deadlocks; FastAPI can still come up
        print("[WARN] MCP port 8000 did not become ready within timeout. Starting FastAPI anyway.")

    # Run FastAPI app
    uvicorn_run(server_module.app, host="0.0.0.0", port=8123)


if __name__ == "__main__":
    main()
