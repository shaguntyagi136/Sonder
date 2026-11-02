"""Application factory. Run with:  uvicorn app.main:create_app --factory   (or: python -m app)"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import router
from app.container import Container, build_container
from app.core.config import Settings, get_settings
from app.core.errors import StorageError
from app.core.logging import setup_logging

log = logging.getLogger(__name__)


def create_app(settings: Optional[Settings] = None, container: Optional[Container] = None) -> FastAPI:
    settings = settings or (container.settings if container else get_settings())
    setup_logging(settings.log_level, settings.app_debug)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("Starting stress-prediction backend v%s with settings: %s", __version__, settings.safe_dump())
        c = container or build_container(settings)
        app.state.container = c
        c.start()
        try:
            yield
        finally:
            c.stop()

    app = FastAPI(title="Stress Prediction IoT Backend", version=__version__, lifespan=lifespan, debug=settings.app_debug)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.include_router(router)

    @app.exception_handler(StorageError)
    async def _storage(request: Request, exc: StorageError):
        log.error("API storage error on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=503, content={"detail": "Storage temporarily unavailable"})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("Unhandled API error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


# Supports the documented ``uvicorn app.main:app --reload`` command while tests
# may still use ``create_app(settings)`` with isolated configuration.
app = create_app()
