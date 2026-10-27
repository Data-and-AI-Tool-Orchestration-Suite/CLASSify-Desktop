"""Add reports.mode column for classification/forecasting switching.

Revision ID: 002
Revises: 001
Create Date: 2026-10-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("reports", sa.Column("mode", sa.String(20), nullable=True))


def downgrade() -> None:
    op.drop_column("reports", "mode")
