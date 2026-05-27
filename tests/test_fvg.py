"""Tests for Fair Value Gap detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.market_structure.fvg import detect_fvg, detect_ifvg
from app.schemas.domain import FVGType


def _make_df(prices):
    rows = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for i, (o, h, l, c) in enumerate(prices):
        rows.append({
            "timestamp": start + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c, "volume": 100.0,
        })
    return pd.DataFrame(rows)


class TestFVGDetection:
    def test_bullish_fvg(self):
        """Candle 0 high < candle 2 low creates a bullish FVG."""
        prices = [
            (100, 102, 99, 101),   # candle 0: high=102
            (101, 120, 100, 119),  # candle 1: impulse candle
            (119, 125, 110, 124),  # candle 2: low=110 > 102 → bullish FVG
            (124, 126, 118, 125),
            (125, 127, 120, 126),
        ]
        df = _make_df(prices)
        fvgs = detect_fvg(df, min_gap_atr_ratio=0.0)

        bullish = [f for f in fvgs if f.fvg_type == FVGType.BULLISH]
        assert len(bullish) >= 1
        fvg = bullish[0]
        assert fvg.bottom == 102.0
        assert fvg.top == 110.0

    def test_bearish_fvg(self):
        """Candle 0 low > candle 2 high creates a bearish FVG."""
        prices = [
            (120, 122, 115, 116),  # candle 0: low=115
            (116, 117, 95, 96),    # candle 1: impulse down
            (96, 105, 90, 100),    # candle 2: high=105 < 115 → bearish FVG
            (100, 102, 95, 97),
            (97, 99, 93, 95),
        ]
        df = _make_df(prices)
        fvgs = detect_fvg(df, min_gap_atr_ratio=0.0)

        bearish = [f for f in fvgs if f.fvg_type == FVGType.BEARISH]
        assert len(bearish) >= 1
        fvg = bearish[0]
        assert fvg.top == 115.0
        assert fvg.bottom == 105.0

    def test_no_fvg_on_overlap(self):
        """Candles that overlap should not produce an FVG."""
        prices = [
            (100, 105, 99, 104),
            (104, 106, 103, 105),
            (105, 107, 104, 106),
            (106, 108, 105, 107),
        ]
        df = _make_df(prices)
        fvgs = detect_fvg(df, min_gap_atr_ratio=0.0)
        assert fvgs == []

    def test_fvg_mitigation(self):
        """A bullish FVG should be marked mitigated when price returns to the zone."""
        prices = [
            (100, 102, 99, 101),
            (101, 120, 100, 119),
            (119, 125, 110, 124),  # bullish FVG: [102, 110]
            (124, 126, 118, 125),
            (125, 126, 101, 102),  # price drops back to 101 < 110 → mitigates
        ]
        df = _make_df(prices)
        fvgs = detect_fvg(df, min_gap_atr_ratio=0.0)

        bullish = [f for f in fvgs if f.fvg_type == FVGType.BULLISH]
        assert len(bullish) >= 1
        assert bullish[0].mitigated is True

    def test_ifvg_flip(self):
        """A mitigated bullish FVG should produce a bearish iFVG."""
        prices = [
            (100, 102, 99, 101),
            (101, 120, 100, 119),
            (119, 125, 110, 124),
            (124, 126, 118, 125),
            (125, 126, 101, 102),
        ]
        df = _make_df(prices)
        fvgs = detect_fvg(df, min_gap_atr_ratio=0.0)
        ifvgs = detect_ifvg(fvgs)

        bearish_ifvg = [f for f in ifvgs if f.fvg_type == FVGType.BEARISH]
        assert len(bearish_ifvg) >= 1

    def test_minimum_data(self):
        """Less than 3 candles should return empty."""
        prices = [(100, 102, 99, 101), (101, 103, 100, 102)]
        df = _make_df(prices)
        assert detect_fvg(df) == []
