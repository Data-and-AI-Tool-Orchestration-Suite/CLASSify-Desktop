"""Initial schema.

Revision ID: 001
Revises:
Create Date: 2026-08-15
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("report_uuid", sa.String(36), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("args", sa.JSON, nullable=True),
        sa.Column("progress", sa.Integer, server_default="0"),
        sa.Column("progress_total", sa.Integer, server_default="0"),
        sa.Column("progress_message", sa.String(255), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "reports",
        sa.Column("uuid", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("original_filename", sa.String(300), nullable=True),
        sa.Column("status", sa.String(50), nullable=False, server_default="Preview"),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("jobs.id"), nullable=True),
        sa.Column("column_changes", sa.JSON, nullable=True),
        sa.Column("comments", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "results",
        sa.Column("uuid", sa.String(36), primary_key=True),
        sa.Column("report_uuid", sa.String(36), sa.ForeignKey("reports.uuid"), nullable=False),
        sa.Column("date_processed", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "settings",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", sa.Text, nullable=False),
    )

    op.create_table(
        "actions",
        sa.Column("uuid", sa.String(36), primary_key=True),
        sa.Column("report_uuid", sa.String(36), sa.ForeignKey("reports.uuid"), nullable=True),
        sa.Column("action", sa.String(255), nullable=False),
        sa.Column("addl_info", sa.JSON, nullable=True),
        sa.Column("time_performed", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("actions")
    op.drop_table("settings")
    op.drop_table("results")
    op.drop_table("reports")
    op.drop_table("jobs")
