"""Add player_crops column to match_stats

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-29
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "match_stats",
        sa.Column("player_crops", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("match_stats", "player_crops")
