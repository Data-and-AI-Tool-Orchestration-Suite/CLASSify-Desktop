"""Tests for the database layer: ORM models, repositories, and migrations."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from classify_api.db import get_engine, get_session_factory, reset_engine, run_migrations
from classify_api.settings import reset_settings


@pytest.fixture()
def db_engine(tmp_data_dir: Path) -> object:
    """Create a fresh DB engine + run migrations for each test."""
    reset_settings()
    reset_engine()
    run_migrations()
    return get_engine()


@pytest.fixture()
def db_session(db_engine: object) -> Session:
    """Yield a DB session with migrations applied."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


class TestMigrations:
    def test_migrations_create_tables(self, db_engine: object) -> None:
        from sqlalchemy import inspect

        inspector = inspect(db_engine)
        tables = set(inspector.get_table_names())
        assert "reports" in tables
        assert "jobs" in tables
        assert "results" in tables
        assert "settings" in tables
        assert "actions" in tables

    def test_migrations_idempotent(self, db_engine: object) -> None:
        # Running again should not error
        run_migrations()


class TestReportRepository:
    def test_create_and_get(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, get_report

        report = create_report(db_session, filename="test.csv")
        db_session.commit()

        fetched = get_report(db_session, report.uuid)
        assert fetched is not None
        assert fetched.filename == "test.csv"
        assert fetched.status == "Preview"

    def test_get_nonexistent(self, db_session: Session) -> None:
        from classify_api.repositories import get_report

        assert get_report(db_session, "nonexistent-uuid") is None

    def test_is_present(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, is_report_present

        assert not is_report_present(db_session, "test.csv")
        create_report(db_session, filename="test.csv")
        db_session.commit()
        assert is_report_present(db_session, "test.csv")

    def test_list_reports(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, list_reports

        for i in range(5):
            create_report(db_session, filename=f"file_{i}.csv")
        db_session.commit()

        reports = list_reports(db_session, start=0, length=10)
        assert len(reports) == 5

    def test_list_reports_with_search(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, list_reports

        create_report(db_session, filename="alpha.csv")
        create_report(db_session, filename="beta.csv")
        db_session.commit()

        results = list_reports(db_session, search="alpha")
        assert len(results) == 1
        assert results[0].filename == "alpha.csv"

    def test_list_reports_pagination(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, list_reports

        for i in range(10):
            create_report(db_session, filename=f"file_{i:02d}.csv")
        db_session.commit()

        page1 = list_reports(db_session, start=0, length=5)
        page2 = list_reports(db_session, start=5, length=5)
        assert len(page1) == 5
        assert len(page2) == 5
        assert page1[0].filename != page2[0].filename

    def test_count_reports(self, db_session: Session) -> None:
        from classify_api.repositories import count_reports, create_report

        assert count_reports(db_session) == 0
        create_report(db_session, filename="a.csv")
        create_report(db_session, filename="b.csv")
        db_session.commit()
        assert count_reports(db_session) == 2

    def test_update_status(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, get_report, update_report_status

        report = create_report(db_session, filename="test.csv")
        db_session.commit()

        update_report_status(db_session, report.uuid, "Processing")
        db_session.commit()

        fetched = get_report(db_session, report.uuid)
        assert fetched is not None
        assert fetched.status == "Processing"

    def test_update_comments(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, get_report, update_report_comments

        report = create_report(db_session, filename="test.csv")
        db_session.commit()

        update_report_comments(db_session, report.uuid, "My notes")
        db_session.commit()

        fetched = get_report(db_session, report.uuid)
        assert fetched is not None
        assert fetched.comments == "My notes"

    def test_delete_report(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, delete_report, get_report

        report = create_report(db_session, filename="test.csv")
        db_session.commit()
        uuid = report.uuid

        assert delete_report(db_session, uuid)
        db_session.commit()
        assert get_report(db_session, uuid) is None

    def test_delete_nonexistent(self, db_session: Session) -> None:
        from classify_api.repositories import delete_report

        assert not delete_report(db_session, "nonexistent")


class TestJobRepository:
    def test_create_and_get(self, db_session: Session) -> None:
        from classify_api.repositories import create_job, create_report, get_job

        report = create_report(db_session, filename="test.csv")
        db_session.flush()

        job = create_job(db_session, report_uuid=report.uuid, args={"model": "rf"})
        db_session.commit()

        fetched = get_job(db_session, job.id)
        assert fetched is not None
        assert fetched.state == "queued"
        assert fetched.args == {"model": "rf"}

    def test_get_next_queued(self, db_session: Session) -> None:
        from classify_api.repositories import create_job, create_report, get_next_queued_job

        report = create_report(db_session, filename="test.csv")
        db_session.flush()

        job1 = create_job(db_session, report_uuid=report.uuid)
        create_job(db_session, report_uuid=report.uuid)
        db_session.commit()

        next_job = get_next_queued_job(db_session)
        assert next_job is not None
        assert next_job.id == job1.id  # FIFO

    def test_update_state(self, db_session: Session) -> None:
        from classify_api.repositories import create_job, create_report, get_job, update_job_state

        report = create_report(db_session, filename="test.csv")
        db_session.flush()
        job = create_job(db_session, report_uuid=report.uuid)
        db_session.commit()

        update_job_state(db_session, job.id, "running")
        db_session.commit()

        fetched = get_job(db_session, job.id)
        assert fetched is not None
        assert fetched.state == "running"
        assert fetched.started_at is not None

    def test_update_progress(self, db_session: Session) -> None:
        from classify_api.repositories import (
            create_job,
            create_report,
            get_job,
            update_job_progress,
        )

        report = create_report(db_session, filename="test.csv")
        db_session.flush()
        job = create_job(db_session, report_uuid=report.uuid)
        db_session.commit()

        update_job_progress(db_session, job.id, completed=5, total=10, message="5/10 Processed")
        db_session.commit()

        fetched = get_job(db_session, job.id)
        assert fetched is not None
        assert fetched.progress == 5
        assert fetched.progress_total == 10
        assert fetched.progress_message == "5/10 Processed"

    def test_finished_states_set_timestamp(self, db_session: Session) -> None:
        from classify_api.repositories import create_job, create_report, get_job, update_job_state

        report = create_report(db_session, filename="test.csv")
        db_session.flush()
        job = create_job(db_session, report_uuid=report.uuid)
        db_session.commit()

        update_job_state(db_session, job.id, "running")
        db_session.commit()
        update_job_state(db_session, job.id, "succeeded")
        db_session.commit()

        fetched = get_job(db_session, job.id)
        assert fetched is not None
        assert fetched.finished_at is not None

    def test_get_running_job(self, db_session: Session) -> None:
        from classify_api.repositories import (
            create_job,
            create_report,
            get_running_job,
            update_job_state,
        )

        report = create_report(db_session, filename="test.csv")
        db_session.flush()
        job = create_job(db_session, report_uuid=report.uuid)
        db_session.commit()

        update_job_state(db_session, job.id, "running")
        db_session.commit()

        running = get_running_job(db_session)
        assert running is not None
        assert running.id == job.id


class TestResultRepository:
    def test_create_and_get(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, create_result, get_result_by_report

        report = create_report(db_session, filename="test.csv")
        db_session.flush()
        create_result(db_session, report_uuid=report.uuid)
        db_session.commit()

        result = get_result_by_report(db_session, report.uuid)
        assert result is not None
        assert result.report_uuid == report.uuid


class TestSettingsRepository:
    def test_get_set_setting(self, db_session: Session) -> None:
        from classify_api.repositories import get_setting, set_setting

        assert get_setting(db_session, "foo") is None
        assert get_setting(db_session, "foo", default="bar") == "bar"

        set_setting(db_session, "foo", "baz")
        db_session.commit()
        assert get_setting(db_session, "foo") == "baz"

    def test_update_existing_setting(self, db_session: Session) -> None:
        from classify_api.repositories import get_setting, set_setting

        set_setting(db_session, "key", "v1")
        set_setting(db_session, "key", "v2")
        db_session.commit()
        assert get_setting(db_session, "key") == "v2"

    def test_seed_defaults(self, db_session: Session) -> None:
        from classify_api.repositories import get_all_settings, seed_default_settings

        seed_default_settings(db_session)
        db_session.commit()

        settings = get_all_settings(db_session)
        assert "n_jobs" in settings
        assert "max_upload_mb" in settings
        assert settings["theme"] == "default"

    def test_seed_defaults_idempotent(self, db_session: Session) -> None:
        from classify_api.repositories import get_setting, seed_default_settings, set_setting

        seed_default_settings(db_session)
        set_setting(db_session, "theme", "dark")
        db_session.commit()

        seed_default_settings(db_session)
        db_session.commit()

        assert get_setting(db_session, "theme") == "dark"  # Not overwritten


class TestSerialization:
    def test_serialize_report(self, db_session: Session) -> None:
        from classify_api.repositories import create_report, serialize_report

        report = create_report(db_session, filename="test.csv", original_filename="orig.csv")
        db_session.flush()

        data = serialize_report(report)
        assert data["filename"] == "test.csv"
        assert data["original_filename"] == "orig.csv"
        assert data["status"] == "Preview"
        assert "uuid" in data
        assert "created_at" in data

    def test_serialize_job(self, db_session: Session) -> None:
        from classify_api.repositories import create_job, create_report, serialize_job

        report = create_report(db_session, filename="test.csv")
        db_session.flush()
        job = create_job(db_session, report_uuid=report.uuid, args={"models": ["rf"]})
        db_session.flush()

        data = serialize_job(job)
        assert data["state"] == "queued"
        assert data["args"] == {"models": ["rf"]}
        assert "id" in data
        assert "created_at" in data
