from __future__ import annotations

import importlib.resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router as api_router
from .settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="arro-server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def _root() -> dict[str, str]:
        return {"service": "arro-server", "version": __version__, "docs": "/docs"}

    if settings.serve_frontend:
        frontend_dir: Path | None = None
        if settings.frontend_dir:
            frontend_dir = Path(settings.frontend_dir)
        else:
            # Development layout: <repo>/frontend/
            _dev = Path(__file__).parent.parent.parent / "frontend"
            if _dev.exists():
                frontend_dir = _dev
            else:
                # Installed wheel: share/arro_server/frontend (hatch shared-data)
                try:
                    _pkg = importlib.resources.files("arro_server") / "../../../share/arro_server/frontend"
                    _resolved = Path(str(_pkg)).resolve()
                    if _resolved.exists():
                        frontend_dir = _resolved
                except Exception:
                    pass
        if frontend_dir and frontend_dir.exists():
            app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

    return app
