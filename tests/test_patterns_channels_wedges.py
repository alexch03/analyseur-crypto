"""Tests pour Channels et Wedges."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.market_structure.swings import detect_swings
from app.patterns.channels import ChannelDetector
from app.patterns.wedges import WedgeDetector
from app.schemas.patterns import BreakoutDirection, PatternKind


def _ohlcv_from_closes(closes: list[float], wick: float = 1.0) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame([
        {
            "timestamp": start + timedelta(hours=i),
            "open": c,
            "high": c + wick,
            "low": c - wick,
            "close": c,
            "volume": 100.0,
        }
        for i, c in enumerate(closes)
    ])


def _channel_up_ohlcv() -> pd.DataFrame:
    """Canal montant : highs et lows montent à pente similaire, écart constant."""
    closes = []
    base = 100.0
    for i in range(40):
        base += 0.5
        # sinusoïde fine autour de la base pour faire des swings
        wave = 4 * (1 if i % 4 < 2 else -1)
        closes.append(base + wave)
    return _ohlcv_from_closes(closes, wick=1.5)


def _channel_down_ohlcv() -> pd.DataFrame:
    closes = []
    base = 200.0
    for i in range(40):
        base -= 0.5
        wave = 4 * (1 if i % 4 < 2 else -1)
        closes.append(base + wave)
    return _ohlcv_from_closes(closes, wick=1.5)


def _interpolated_pivots(
    pivot_lows: list[tuple[int, float]],
    pivot_highs: list[tuple[int, float]],
    n: int,
) -> list[float]:
    """Interpole linéairement les closes entre pivots alternés bas/haut."""
    all_pivots = sorted(pivot_lows + pivot_highs, key=lambda x: x[0])
    closes = [0.0] * n
    for idx, price in all_pivots:
        closes[idx] = price
    for k in range(len(all_pivots) - 1):
        i0, p0 = all_pivots[k]
        i1, p1 = all_pivots[k + 1]
        for j in range(i0 + 1, i1):
            t = (j - i0) / (i1 - i0)
            closes[j] = p0 + (p1 - p0) * t
    # Avant/après les pivots externes : extrapole platement
    first_i, first_p = all_pivots[0]
    last_i, last_p = all_pivots[-1]
    for j in range(0, first_i):
        closes[j] = first_p
    for j in range(last_i + 1, n):
        closes[j] = last_p
    return closes


def _rising_wedge_ohlcv() -> pd.DataFrame:
    """Rising wedge : highs +2 sur 24 bars, lows +10 sur 24 bars (apex en avant)."""
    n = 30
    lows = [(0, 100.0), (6, 102.5), (12, 105.0), (18, 107.5), (24, 110.0)]
    highs = [(3, 111.0), (9, 111.5), (15, 112.0), (21, 112.5), (27, 113.0)]
    closes = _interpolated_pivots(lows, highs, n)
    return _ohlcv_from_closes(closes, wick=0.3)


def _falling_wedge_ohlcv() -> pd.DataFrame:
    """Falling wedge : highs −10 sur 24 bars, lows −4 sur 24 bars (convergence)."""
    n = 30
    highs = [(0, 200.0), (6, 197.5), (12, 195.0), (18, 192.5), (24, 190.0)]
    lows = [(3, 189.0), (9, 188.0), (15, 187.0), (21, 186.0), (27, 185.0)]
    closes = _interpolated_pivots(lows, highs, n)
    return _ohlcv_from_closes(closes, wick=0.3)


def test_channel_up_detected():
    df = _channel_up_ohlcv()
    swings = detect_swings(df, left=2, right=2)
    patterns = ChannelDetector(min_pivots_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h"
    )
    assert len(patterns) == 1
    p = patterns[0]
    assert p.kind == PatternKind.CHANNEL_UP
    assert p.breakout_direction == BreakoutDirection.UP
    assert p.target is not None and p.target > p.breakout_level


def test_channel_down_detected():
    df = _channel_down_ohlcv()
    swings = detect_swings(df, left=2, right=2)
    patterns = ChannelDetector(min_pivots_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h"
    )
    assert len(patterns) == 1
    assert patterns[0].kind == PatternKind.CHANNEL_DOWN
    assert patterns[0].breakout_direction == BreakoutDirection.DOWN


def test_no_channel_on_flat_market():
    df = _ohlcv_from_closes([100.0 + (1 if i % 2 else -1) for i in range(30)])
    swings = detect_swings(df, left=2, right=2)
    patterns = ChannelDetector(min_pivots_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h"
    )
    assert patterns == []


def test_rising_wedge_detected():
    df = _rising_wedge_ohlcv()
    swings = detect_swings(df, left=2, right=2)
    patterns = WedgeDetector(min_pivots_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h"
    )
    assert len(patterns) >= 1, f"got: {patterns}"
    p = patterns[0]
    assert p.kind == PatternKind.WEDGE_RISING
    assert p.breakout_direction == BreakoutDirection.DOWN
    assert p.target is not None and p.target < p.breakout_level


def test_falling_wedge_detected():
    df = _falling_wedge_ohlcv()
    swings = detect_swings(df, left=2, right=2)
    patterns = WedgeDetector(min_pivots_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h"
    )
    assert len(patterns) >= 1
    p = patterns[0]
    assert p.kind == PatternKind.WEDGE_FALLING
    assert p.breakout_direction == BreakoutDirection.UP
    assert p.target is not None and p.target > p.breakout_level


def test_wedge_does_not_match_channel():
    """Un wedge convergent ne doit pas matcher un channel parallèle."""
    df = _rising_wedge_ohlcv()
    swings = detect_swings(df, left=2, right=2)
    ch = ChannelDetector(min_pivots_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h"
    )
    assert ch == [], "rising wedge ne doit pas être détecté comme channel"
