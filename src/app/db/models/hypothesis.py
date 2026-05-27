"""ORM models pour les hypothèses de patterns chartistes + unit trades.

Séparé de ``candle.py`` pour limiter la taille du fichier original et faciliter
les recherches. Partage la même ``Base`` déclarative.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.candle import Base


class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    timeframe_id: Mapped[int] = mapped_column(ForeignKey("timeframes.id"), nullable=False)
    pattern_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    target_price: Mapped[float] = mapped_column(Float, nullable=False)
    invalidation_price: Mapped[float] = mapped_column(Float, nullable=False)
    triggered_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    confluence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    arm_proximity_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.005)
    expiry_bars: Mapped[int] = mapped_column(Integer, nullable=False, default=40)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    triggered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pattern_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    transitions: Mapped[list | None] = mapped_column(JSON, nullable=True)
    confluence_tags: Mapped[list | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_hypothesis_symbol_state", "symbol_id", "state"),
        Index("ix_hypothesis_active_lookup", "symbol_id", "timeframe_id", "state", "pattern_kind"),
    )


class UnitTrade(Base):
    __tablename__ = "unit_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    hypothesis_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("hypotheses.id"), nullable=False, index=True
    )
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    timeframe_id: Mapped[int] = mapped_column(ForeignKey("timeframes.id"), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    pattern_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    entry_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pct_gain: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(24), nullable=True)
    confluence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    confluence_tags: Mapped[list | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_unit_trade_symbol_time", "symbol_id", "entry_timestamp"),
        Index("ix_unit_trade_outcome", "outcome"),
    )


class ScanRun(Base):
    """Trace des passes du scanner continu : utile pour debug et UI (dernier scan ok).

    Un row par (symbol, timeframe) à chaque cycle terminé. On garde les N derniers
    via un job de purge si besoin.
    """
    __tablename__ = "scan_runs"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=True)
    symbol_id: Mapped[int] = mapped_column(ForeignKey("symbols.id"), nullable=False)
    timeframe_id: Mapped[int] = mapped_column(ForeignKey("timeframes.id"), nullable=False)
    ts_started: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ts_finished: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    candles_fetched: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    patterns_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    hypotheses_active: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)

    __table_args__ = (
        Index("ix_scan_run_lookup", "symbol_id", "timeframe_id", "ts_finished"),
    )
