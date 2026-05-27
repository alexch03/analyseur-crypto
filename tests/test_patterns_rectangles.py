"""Tests pour le détecteur de rectangles."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from app.market_structure.swings import detect_swings
from app.patterns.rectangles import RectangleDetector
from app.schemas.patterns import BreakoutDirection, PatternKind


def _make_ohlcv(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return pd.DataFrame(
        [
            {
                "timestamp": start + timedelta(hours=i),
                "open": o, "high": h, "low": l, "close": c, "volume": 100.0,
            }
            for i, (o, h, l, c) in enumerate(prices)
        ]
    )


def test_rectangle_detected_with_clear_box() -> None:
    """3 touches ~110 et 3 touches ~90, range >2%, close au milieu."""
    prices = [
        (100, 105, 99, 104),
        (104, 109, 103, 108),
        (108, 110.0, 105, 107),    # H 110
        (107, 108, 95, 96),
        (96, 97, 90.0, 91),        # L 90
        (91, 98, 91, 97),
        (97, 108, 97, 107),
        (107, 110.2, 105, 106),    # H 110.2
        (106, 107, 95, 96),
        (96, 97, 90.1, 91),        # L 90.1
        (91, 99, 91, 98),
        (98, 108, 97, 107),
        (107, 109.9, 105, 106),    # H 109.9
        (106, 107, 96, 96),
        (96, 97, 90.0, 91),        # L 90.0
        (91, 99, 91, 98),
        (98, 100, 97, 99),
        (99, 102, 98, 100),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = RectangleDetector(min_touches_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h",
    )
    assert len(patterns) == 1
    p = patterns[0]
    assert p.kind == PatternKind.RECTANGLE
    assert p.breakout_direction == BreakoutDirection.UNDETERMINED
    assert 109.5 <= p.breakout_level <= 110.5
    assert 89.5 <= p.invalidation_level <= 90.5
    assert p.height > 0
    assert p.payload["resistance_touches"] >= 3
    assert p.payload["support_touches"] >= 3


def test_no_rectangle_when_too_few_touches() -> None:
    prices = [
        (100, 110, 99, 109),
        (109, 110, 95, 96),
        (96, 97, 90, 91),
        (91, 100, 91, 99),
        (99, 101, 95, 96),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = RectangleDetector(min_touches_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h",
    )
    assert patterns == []


def test_no_rectangle_when_range_too_narrow() -> None:
    """Range < min_range_pct (2%) → pas un vrai rectangle."""
    prices = [
        (100, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
        (100.5, 101.0, 99.5, 100.5),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = RectangleDetector(min_touches_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h",
    )
    assert patterns == []


def test_no_rectangle_when_already_broken_above() -> None:
    """Une cassure largement haussière invalide le rectangle (close >> résistance)."""
    prices = [
        (100, 105, 99, 104),
        (104, 109, 103, 108),
        (108, 110.0, 105, 107),
        (107, 108, 95, 96),
        (96, 97, 90.0, 91),
        (91, 98, 91, 97),
        (97, 108, 97, 107),
        (107, 110.2, 105, 106),
        (106, 107, 95, 96),
        (96, 97, 90.1, 91),
        (91, 99, 91, 98),
        (98, 108, 97, 107),
        (107, 109.9, 105, 106),
        (106, 107, 96, 96),
        (96, 97, 90.0, 91),
        (91, 99, 91, 98),
        (98, 120, 98, 119),        # cassure violente vers le haut
        (119, 125, 118, 124),
    ]
    df = _make_ohlcv(prices)
    swings = detect_swings(df, left=2, right=2)
    patterns = RectangleDetector(min_touches_per_side=3).detect(
        df, swings, symbol="TEST/USDT", timeframe="1h",
    )
    assert patterns == []
