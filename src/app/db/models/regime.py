"""Historisation du regime de marche detecte au fil du temps."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, String, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.candle import Base


class MarketRegimeSnapshot(Base):
    """Snapshot du regime detecte (une ligne toutes les N minutes)."""
    __tablename__ = "market_regime_snapshots"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True, autoincrement=True,
    )
    snapshot_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True,
    )
    trend: Mapped[str] = mapped_column(String(8), nullable=False)  # BULL/BEAR/RANGE
    volatility: Mapped[str] = mapped_column(String(8), nullable=False)  # LOW/NORMAL/HIGH
    strength: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    btc_change_24h_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    btc_above_sma50: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    btc_above_sma200: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    breadth_pct: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    atr_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        Index("ix_regime_lookup", "snapshot_ts", "trend"),
    )
