"""Tracker du regime de marche : detecte, persiste, expose.

Usage typique :
    tracker = get_regime_tracker()
    await tracker.refresh(session, btc_ohlcv, breadth_pct=...)
    regime = tracker.current()  # MarketRegime ou None
    # ... HypothesisEngine utilise tracker.current() pour adapter ses filtres

Refresh appele depuis le scanner toutes les N minutes (typiquement chaque cycle).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MarketRegimeSnapshot
from app.services.market_regime import MarketRegime, detect_regime

logger = logging.getLogger(__name__)


class RegimeTracker:
    """Singleton qui maintient le regime courant + historique recent."""

    def __init__(self) -> None:
        self._current: MarketRegime | None = None
        self._last_snapshot_ts: datetime | None = None
        self._min_interval_seconds: int = 60  # ne snapshot pas plus que toutes les 60s

    def current(self) -> MarketRegime | None:
        return self._current

    async def refresh(
        self,
        session: AsyncSession,
        btc_ohlcv: pd.DataFrame,
        *,
        breadth_pct: float | None = None,
        force_snapshot: bool = False,
    ) -> MarketRegime:
        """Detecte le regime courant et le persiste si change ou si delay > min_interval."""
        regime = detect_regime(btc_ohlcv, breadth_pct=breadth_pct)
        self._current = regime

        now = datetime.now(timezone.utc)
        should_save = force_snapshot
        if not should_save:
            if self._last_snapshot_ts is None:
                should_save = True
            elif (now - self._last_snapshot_ts).total_seconds() >= self._min_interval_seconds:
                should_save = True

        if should_save:
            try:
                snap = MarketRegimeSnapshot(
                    snapshot_ts=now,
                    trend=regime.trend,
                    volatility=regime.volatility,
                    strength=regime.strength,
                    btc_change_24h_pct=regime.btc_change_24h_pct,
                    btc_above_sma50=regime.btc_above_sma50,
                    btc_above_sma200=regime.btc_above_sma200,
                    breadth_pct=regime.breadth_pct,
                    atr_pct=regime.atr_pct,
                )
                session.add(snap)
                await session.flush()
                self._last_snapshot_ts = now
                logger.info(
                    "Regime snapshot: %s %s strength=%.2f btc_24h=%.2f%% breadth=%.0f%% atr=%.2f%%",
                    regime.trend, regime.volatility, regime.strength,
                    regime.btc_change_24h_pct, regime.breadth_pct, regime.atr_pct,
                )
            except Exception:
                logger.exception("Failed to snapshot regime")
        return regime

    async def load_latest_from_db(self, session: AsyncSession) -> MarketRegime | None:
        """Au demarrage : charge le dernier snapshot persiste pour bootstrap."""
        q = (select(MarketRegimeSnapshot)
              .order_by(MarketRegimeSnapshot.snapshot_ts.desc())
              .limit(1))
        row = (await session.execute(q)).scalar_one_or_none()
        if row is None:
            return None
        self._current = MarketRegime(
            trend=row.trend,
            volatility=row.volatility,
            strength=float(row.strength),
            btc_change_24h_pct=float(row.btc_change_24h_pct),
            btc_above_sma50=bool(row.btc_above_sma50),
            btc_above_sma200=bool(row.btc_above_sma200),
            breadth_pct=float(row.breadth_pct),
            atr_pct=float(row.atr_pct),
            detected_at=row.snapshot_ts,
        )
        return self._current

    async def fetch_history(
        self, session: AsyncSession, limit: int = 100,
    ) -> list[dict]:
        """Retourne l'historique recent des snapshots (pour dashboard chart)."""
        q = (select(MarketRegimeSnapshot)
              .order_by(MarketRegimeSnapshot.snapshot_ts.desc())
              .limit(limit))
        rows = (await session.execute(q)).scalars().all()
        return [
            {
                "ts": r.snapshot_ts.isoformat(),
                "trend": r.trend,
                "volatility": r.volatility,
                "strength": float(r.strength),
                "btc_change_24h_pct": float(r.btc_change_24h_pct),
                "breadth_pct": float(r.breadth_pct),
                "atr_pct": float(r.atr_pct),
            }
            for r in rows
        ]


# Singleton
_tracker: RegimeTracker | None = None


def get_regime_tracker() -> RegimeTracker:
    global _tracker
    if _tracker is None:
        _tracker = RegimeTracker()
    return _tracker
