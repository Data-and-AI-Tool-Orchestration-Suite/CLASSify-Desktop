"""SQLAlchemy 2.0 database engine, session factory, and FastAPI dependency."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from classify_api.settings import get_settings

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Return the cached SQLAlchemy engine (created on first call)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        db_path = settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        _engine = create_engine(
            url,
            echo=False,
            connect_args={"check_same_thread": False},
        )

        # Enable WAL mode for better read/write concurrency
        @event.listens_for(_engine, "connect")
        def _set_wal(dbapi_conn: object, _record: object) -> None:
            cursor = dbapi_conn.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the cached session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


def get_session() -> Iterator[Session]:
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for scripts/tests that need a session outside FastAPI."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine() -> None:
    """Clear cached engine + session factory (used by tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def run_migrations() -> None:
    """Run Alembic migrations to bring the DB to the latest schema.

    Called on app startup.  Uses the programmatic Alembic API.
    """
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    migrations_dir = Path(__file__).resolve().parent.parent / "migrations"
    alembic_cfg = Config(str(migrations_dir / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(migrations_dir))

    settings = get_settings()
    settings.ensure_dirs()
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{settings.db_path}")

    command.upgrade(alembic_cfg, "head")
