"""Tests pour le détecteur de flags (bull/bear)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.market_structure.swings import detect_swings
from app.patterns.flags import FlagDetector
from app.schemas.patterns import BreakoutDirection, PatternKind


def _df(closes: list[float], wick: float = 0.5) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame([
        {
            "timestamp": start + timedelta(hours=i),
            "open": c, "high": c + wick, "low": c - wick, "close": c, "volume": 100.0,
        }
        for i, c in enumerate(closes)
    ])


def test_bull_flag_detected():
    """Pole +10% sur 6 bars, puis flag 8 bars en range serré juste sous le top."""
    closes = [100.0] * 6
    # Pole : 100 → 110 sur 6 bars
    for i in range(1, 7):
        closes.append(100 + i * 10 / 6)
    # Flag : range entre 108 et 110, 10 bars
    flag_pattern = [109.5, 108.5, 109.0, 108.5, 109.5, 109.0, 108.5, 109.2, 109.5, 109.5]
    closes.extend(flag_pattern)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = FlagDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert len(p) == 1
    assert p[0].kind == PatternKind.FLAG_BULL
    assert p[0].breakout_direction == BreakoutDirection.UP
    assert p[0].target is not None and p[0].target > p[0].breakout_level
    assert 108 <= p[0].invalidation_level <= 109.5
    assert p[0].height > 8.0   # ~10


def test_bear_flag_detected():
    """Pole -10% puis consolidation latérale juste au-dessus du bas du pole."""
    closes = [200.0] * 6
    for i in range(1, 7):
        closes.append(200 - i * 20 / 6)
    flag_pattern = [183.0, 184.5, 183.5, 184.0, 183.0, 184.0, 183.5, 184.2, 183.0, 183.5]
    closes.extend(flag_pattern)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = FlagDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert len(p) == 1
    assert p[0].kind == PatternKind.FLAG_BEAR
    assert p[0].breakout_direction == BreakoutDirection.DOWN
    assert p[0].target is not None and p[0].target < p[0].breakout_level


def test_no_flag_without_pole():
    """Marché plat → pas de pole → pas de flag."""
    closes = [100 + (1 if i % 2 == 0 else -1) for i in range(30)]
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = FlagDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert p == []


def test_no_flag_when_consolidation_retraces_too_much():
    """Pole +10% puis retracement à 70% (au-delà des 55% autorisés) → pas un flag."""
    closes = [100.0] * 6
    for i in range(1, 7):
        closes.append(100 + i * 10 / 6)
    # Retracement à 103 ≈ 70% du pole (110→100)
    deep_retrace = [108, 106, 103, 104, 103, 104, 103, 103, 104, 103]
    closes.extend(deep_retrace)
    df = _df(closes)
    swings = detect_swings(df, left=2, right=2)
    p = FlagDetector().detect(df, swings, symbol="TEST/USDT", timeframe="1h")
    assert p == []
