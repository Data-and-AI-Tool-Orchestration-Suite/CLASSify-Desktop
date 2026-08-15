"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from classify_api.routers import addons, datasets, jobs, results, system
from classify_api.settings import get_settings

log = structlog.get_logger()


def _configure_logging(dev: bool) -> None:
    """Configure structlog + stdlib logging."""
    logging.basicConfig(
        level=logging.DEBUG if dev else logging.INFO,
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            (structlog.dev.ConsoleRenderer() if dev else structlog.processors.JSONRenderer()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if dev else logging.INFO),
        cache_logger_on_first_use=True,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    _configure_logging(settings.dev_mode)
    log.info(
        "classify_api.starting",
        data_dir=str(settings.data_dir),
        dev=settings.dev_mode,
    )

    # Run database migrations
    from classify_api.db import run_migrations

    run_migrations()

    # Recover stale jobs + seed settings
    from classify_api.db import get_session_factory
    from classify_api.repositories import seed_default_settings
    from runner.queue import mark_stale_jobs_failed

    db_factory = get_session_factory()
    db = db_factory()
    try:
        seed_default_settings(db)
        db.commit()
        recovered = mark_stale_jobs_failed(db)
        db.commit()
        if recovered:
            log.info("classify_api.recovered_jobs", count=recovered)
    finally:
        db.close()

    # Start the job manager
    from runner.manager import start_manager

    start_manager()

    # Initialize add-ons (prepend sys.path for installed packages)
    from classify_api.services.addon_service import init_addons

    init_addons()

    yield

    # Stop the job manager
    from runner.manager import stop_manager

    stop_manager()
    log.info("classify_api.stopping")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="CLASSify Desktop API",
        version="1.0.0.dev0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.dev_mode else None,
        openapi_url="/openapi.json" if settings.dev_mode else None,
    )

    # CORS is off for production local use; enabled in dev for the Vite proxy.
    if settings.dev_mode:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ── Routers ──
    app.include_router(system.router, prefix="/api/system", tags=["system"])
    app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
    app.include_router(results.router, prefix="/api/results", tags=["results"])
    app.include_router(addons.router, prefix="/api/addons", tags=["addons"])

    # ── SPA static (served from frontend/dist when available) ──
    # Look for the built frontend relative to the package location,
    # not the data directory. In a frozen app this is the bundle root.
    import sys
    from pathlib import Path

    if getattr(sys, "frozen", False):
        spa_dist = Path(sys._MEIPASS) / "frontend" / "dist"  # type: ignore[attr-defined]  # noqa: SLF001
    else:
        # backend/classify_api/main.py → repo root = 3 levels up
        spa_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"

    if spa_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(spa_dist), html=True), name="spa")

    return app


app = create_app()


def run() -> None:
    """Entry point for the ``classify-api`` console script."""
    settings = get_settings()
    uvicorn.run(
        "classify_api.main:app",
        host=settings.host,
        port=settings.port or 8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
