"""hypotheses + unit_trades + scan_runs

Tables pour le scanner continu de patterns chartistes + tracker paper unit-based.

Revision ID: a3b1c2d4e5f6
Revises: e8c2aea87d0c
Create Date: 2026-05-27 12:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a3b1c2d4e5f6"
down_revision: Union[str, None] = "e8c2aea87d0c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "hypotheses",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("timeframe_id", sa.Integer(), sa.ForeignKey("timeframes.id"), nullable=False),
        sa.Column("pattern_kind", sa.String(length=40), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False, index=True),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("target_price", sa.Float(), nullable=False),
        sa.Column("invalidation_price", sa.Float(), nullable=False),
        sa.Column("triggered_price", sa.Float(), nullable=True),
        sa.Column("outcome_price", sa.Float(), nullable=True),
        sa.Column("confluence_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("arm_proximity_pct", sa.Float(), nullable=False, server_default="0.005"),
        sa.Column("expiry_bars", sa.Integer(), nullable=False, server_default="40"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("triggered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pattern_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("transitions", postgresql.JSONB(), nullable=True),
        sa.Column("confluence_tags", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_hypothesis_symbol_state", "hypotheses", ["symbol_id", "state"])
    op.create_index(
        "ix_hypothesis_active_lookup",
        "hypotheses",
        ["symbol_id", "timeframe_id", "state", "pattern_kind"],
    )

    op.create_table(
        "unit_trades",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "hypothesis_id",
            sa.String(length=36),
            sa.ForeignKey("hypotheses.id"),
            nullable=False,
            index=True,
        ),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("timeframe_id", sa.Integer(), sa.ForeignKey("timeframes.id"), nullable=False),
        sa.Column("side", sa.String(length=8), nullable=False),
        sa.Column("pattern_kind", sa.String(length=40), nullable=False),
        sa.Column("entry_price", sa.Float(), nullable=False),
        sa.Column("entry_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("exit_price", sa.Float(), nullable=True),
        sa.Column("exit_timestamp", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pct_gain", sa.Float(), nullable=True),
        sa.Column("outcome", sa.String(length=24), nullable=True),
        sa.Column("confluence_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("confluence_tags", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_unit_trade_symbol_time", "unit_trades", ["symbol_id", "entry_timestamp"])
    op.create_index("ix_unit_trade_outcome", "unit_trades", ["outcome"])

    op.create_table(
        "scan_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("symbol_id", sa.Integer(), sa.ForeignKey("symbols.id"), nullable=False),
        sa.Column("timeframe_id", sa.Integer(), sa.ForeignKey("timeframes.id"), nullable=False),
        sa.Column("ts_started", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ts_finished", sa.DateTime(timezone=True), nullable=False),
        sa.Column("candles_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("patterns_detected", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("hypotheses_active", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.String(length=500), nullable=True),
    )
    op.create_index(
        "ix_scan_run_lookup", "scan_runs", ["symbol_id", "timeframe_id", "ts_finished"]
    )


def downgrade() -> None:
    op.drop_index("ix_scan_run_lookup", table_name="scan_runs")
    op.drop_table("scan_runs")
    op.drop_index("ix_unit_trade_outcome", table_name="unit_trades")
    op.drop_index("ix_unit_trade_symbol_time", table_name="unit_trades")
    op.drop_table("unit_trades")
    op.drop_index("ix_hypothesis_active_lookup", table_name="hypotheses")
    op.drop_index("ix_hypothesis_symbol_state", table_name="hypotheses")
    op.drop_table("hypotheses")
