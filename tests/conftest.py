"""Shared test fixtures with small synthetic OHLCV datasets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest


def _make_ohlcv(prices: list[tuple[float, float, float, float]], start: datetime | None = None):
    """Build an OHLCV DataFrame from a list of (open, high, low, close) tuples."""
    if start is None:
        start = datetime(2025, 1, 1, tzinfo=UTC)
    rows = []
    for i, (o, h, l, c) in enumerate(prices):
        rows.append(
            {
                "timestamp": start + timedelta(hours=i),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 100.0,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_ohlcv_bull():
    """Simple bullish trend: higher highs, higher lows.

    Bar indices 0..9  —  clear uptrend with swings at indices 2 (LOW), 4 (HIGH),
    6 (LOW), 8 (HIGH).
    """
    prices = [
        # idx 0        1         2(HL)     3         4(HH)
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 103, 97, 98),   # swing low at 97
        (98, 105, 98, 104),
        (104, 110, 103, 109),  # swing high at 110
        (109, 109, 104, 105),
        (105, 106, 100, 101),  # swing low at 100 (higher than 97)
        (101, 108, 101, 107),
        (107, 115, 106, 114),  # swing high at 115 (higher than 110)
        (114, 114, 110, 112),
    ]
    return _make_ohlcv(prices)


@pytest.fixture()
def tiny_ohlcv_range():
    """Sideways/range market: swing highs and lows cluster around similar prices.

    Designed so that swing highs cluster near 110 and swing lows near 90.
    """
    prices = [
        (100, 102, 98, 101),
        (101, 105, 99, 100),
        (100, 100, 89, 91),   # low ~89
        (91, 109, 90, 108),
        (108, 111, 105, 109),  # high ~111
        (109, 108, 100, 101),
        (101, 102, 90, 92),   # low ~90
        (92, 110, 91, 108),
        (108, 112, 106, 110),  # high ~112
        (110, 111, 105, 107),
    ]
    return _make_ohlcv(prices)


@pytest.fixture()
def tiny_ohlcv_choch():
    """Bullish trend followed by a CHOCH (lower low breaking the sequence).

    Bars 0..11 — uptrend from 0..7, then a lower-low violation at bar 9.
    """
    prices = [
        (100, 102, 99, 101),   # 0
        (101, 103, 100, 102),  # 1
        (102, 103, 96, 97),    # 2  swing low 96
        (97, 106, 97, 105),    # 3
        (105, 112, 104, 111),  # 4  swing high 112
        (111, 111, 106, 107),  # 5
        (107, 108, 99, 100),   # 6  swing low 99 (higher than 96)
        (100, 114, 100, 113),  # 7
        (113, 116, 112, 115),  # 8  swing high 116
        (115, 115, 94, 95),    # 9  swing low 94 — LOWER than 99 → CHOCH
        (95, 100, 95, 97),     # 10  low 95 > 94
        (97, 98, 96, 91),      # 11  low 96 > 94
    ]
    return _make_ohlcv(prices)
