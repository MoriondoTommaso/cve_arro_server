from __future__ import annotations

import importlib.resources
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router as api_router
from .settings import Settings, get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Warm up optional services at startup.

    All three services are optional — if their data directories or
    dependencies are missing the server still starts; endpoints that need
    them will return 503 on first request.
    """
    # --- warm EmbedderService ---
    try:
        from .embedder import EmbedderService
        EmbedderService.get()
        log.info("EmbedderService warmed up successfully")
    except Exception as exc:
        log.warning("EmbedderService warm-up skipped: %s", exc)

    # --- warm PromptSearchEngine ---
    try:
        from .search_engine import PromptSearchEngine
        PromptSearchEngine.get()
        log.info("PromptSearchEngine warmed up successfully")
    except Exception as exc:
        log.warning("PromptSearchEngine warm-up skipped: %s", exc)

    # --- warm CveDriftEngine ---
    try:
        from .drift_engine import CveDriftEngine
        CveDriftEngine.get()
        log.info("CveDriftEngine warmed up successfully")
    except Exception as exc:
        log.warning("CveDriftEngine warm-up skipped: %s", exc)

    yield  # application is now running


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="arro-server",
        version=__version__,
        description="Serve Zarr v3 datasets and ArrowSpace metadata over HTTP.",
        lifespan=_lifespan,
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
            _dev = Path(__file__).parent.parent.parent / "frontend"
            if _dev.exists():
                frontend_dir = _dev
            else:
                try:
                    _pkg = (
                        importlib.resources.files("arro_server")
                        / "../../../share/arro_server/frontend"
                    )
                    _resolved = Path(str(_pkg)).resolve()
                    if _resolved.exists():
                        frontend_dir = _resolved
                except Exception:
                    pass
        if frontend_dir and frontend_dir.exists():
            @app.get("/ui", include_in_schema=False)
            def _ui_redirect() -> RedirectResponse:
                return RedirectResponse(url="/ui/", status_code=307)

            app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")

    return app
