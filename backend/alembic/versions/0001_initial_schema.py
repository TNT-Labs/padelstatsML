"""Initial schema: matches + match_stats

Revision ID: 0001
Revises:
Create Date: 2026-04-28
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# create_type=False: SQLAlchemy non emette CREATE TYPE durante op.create_table;
# lo gestiamo noi esplicitamente tramite DO block (idempotente).
matchstatus = sa.Enum(
    "uploading", "queued", "processing", "completed", "failed",
    name="matchstatus",
    create_type=False,
)


def upgrade() -> None:
    # CREATE TYPE IF NOT EXISTS via PL/pgSQL — idempotente al 100%
    op.execute(sa.text(
        "DO $$ BEGIN "
        "  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'matchstatus') "
        "  THEN CREATE TYPE matchstatus AS ENUM "
        "    ('uploading', 'queued', 'processing', 'completed', 'failed'); "
        "  END IF; "
        "END $$"
    ))

    op.create_table(
        "matches",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column(
            "status",
            matchstatus,
            nullable=False,
            server_default="uploading",
        ),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error_message", sa.String(1000), nullable=True),
        sa.Column("video_s3_key", sa.String(500), nullable=False),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("player_names", JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_matches_status", "matches", ["status"])
    op.create_index("ix_matches_created_at", "matches", ["created_at"])

    op.create_table(
        "match_stats",
        sa.Column(
            "match_id",
            UUID(as_uuid=True),
            sa.ForeignKey("matches.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("per_player", JSONB, nullable=False),
        sa.Column("heatmaps", JSONB, nullable=False),
        sa.Column("rallies", JSONB, nullable=False),
        sa.Column("court_calibration", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table("match_stats")
    op.drop_index("ix_matches_created_at", table_name="matches")
    op.drop_index("ix_matches_status", table_name="matches")
    op.drop_table("matches")
    op.execute(sa.text("DROP TYPE IF EXISTS matchstatus"))
