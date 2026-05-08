from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    host = os.environ.get("ARRO_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ARRO_SERVER_PORT", "8000"))
    reload = os.environ.get("ARRO_SERVER_RELOAD", "0") == "1"
    uvicorn.run(
        "arro_server.app:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    main()
