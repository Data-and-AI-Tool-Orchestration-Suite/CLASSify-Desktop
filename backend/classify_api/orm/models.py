"""SQLAlchemy ORM models for the CLASSify Desktop database.

Simplified from the web app's Postgres schema — no users, tenants, or
plugins.  Single-user, local-only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class Report(Base):
    """A uploaded dataset and its processing state."""

    __tablename__ = "reports"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(300), nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="Preview")
    job_id: Mapped[str | None] = mapped_column(String(36), ForeignKey("jobs.id"), nullable=True)
    column_changes: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    comments: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job: Mapped[Job | None] = relationship(foreign_keys=[job_id])
    results: Mapped[list[Result]] = relationship(
        back_populates="report", cascade="all, delete-orphan"
    )


class Job(Base):
    """A training job (replaces ClearML tasks)."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    report_uuid: Mapped[str] = mapped_column(String(36), ForeignKey("reports.uuid"), nullable=False)
    state: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    args: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    progress_message: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    report: Mapped[Report] = relationship(foreign_keys=[report_uuid])

    # Valid states: queued, running, cancelling, succeeded, failed


class Result(Base):
    """A completed training result (one per successful job)."""

    __tablename__ = "results"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    report_uuid: Mapped[str] = mapped_column(String(36), ForeignKey("reports.uuid"), nullable=False)
    date_processed: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    report: Mapped[Report] = relationship(back_populates="results")


class Setting(Base):
    """Application settings (key-value store)."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class Action(Base):
    """Local action history (optional audit trail)."""

    __tablename__ = "actions"

    uuid: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    report_uuid: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("reports.uuid"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    addl_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    time_performed: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
