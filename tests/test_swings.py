"""Tests for swing point (fractal pivot) detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.market_structure.swings import detect_swings
from app.schemas.domain import SwingKind


class TestDetectSwings:
    def test_bullish_trend_swings(self, tiny_ohlcv_bull):
        swings = detect_swings(tiny_ohlcv_bull, left=2, right=2)

        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        lows = [s for s in swings if s.kind == SwingKind.LOW]

        assert len(lows) >= 1, "Should detect at least one swing low"
        assert len(highs) >= 1, "Should detect at least one swing high"

        high_prices = [h.price for h in highs]
        low_prices = [l.price for l in lows]

        assert all(h > max(low_prices) for h in high_prices), "Highs should be above lows"

    def test_range_swings(self, tiny_ohlcv_range):
        swings = detect_swings(tiny_ohlcv_range, left=2, right=2)
        assert len(swings) >= 2, "Range market should produce at least 2 swings"

    def test_minimum_data_returns_empty(self):
        """Fewer bars than left+right+1 should return no swings."""
        prices = [(100, 102, 99, 101), (101, 103, 100, 102)]
        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                    "open": o, "high": h, "low": l, "close": c, "volume": 10.0,
                }
                for i, (o, h, l, c) in enumerate(prices)
            ]
        )
        assert detect_swings(df, left=2, right=2) == []

    def test_plateau_no_swing(self):
        """A flat series (all same high/low) should produce no swings because
        strict left comparison fails."""
        n = 10
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i) for i in range(n)],
                "open": [100.0] * n,
                "high": [100.0] * n,
                "low": [100.0] * n,
                "close": [100.0] * n,
                "volume": [10.0] * n,
            }
        )
        swings = detect_swings(df, left=2, right=2)
        assert swings == [], "Flat series should yield no swings (strict left comparison)"

    def test_single_peak(self):
        """V-shape: one clear high in the middle."""
        prices = [
            (10, 11, 9, 10),
            (10, 12, 10, 11),
            (11, 20, 11, 19),   # clear peak
            (19, 19, 12, 13),
            (13, 14, 10, 11),
        ]
        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                    "open": o, "high": h, "low": l, "close": c, "volume": 10.0,
                }
                for i, (o, h, l, c) in enumerate(prices)
            ]
        )
        swings = detect_swings(df, left=2, right=2)
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        assert len(highs) == 1
        assert highs[0].price == 20.0
        assert highs[0].index == 2

    def test_tie_break_right_side(self):
        """When the right side has an equal high, the left bar should still
        be detected as a swing high (>= on the right)."""
        prices = [
            (10, 11, 9, 10),
            (10, 12, 10, 11),
            (11, 15, 11, 14),   # candidate peak
            (14, 15, 12, 13),   # equal high on right
            (13, 14, 10, 11),
        ]
        df = pd.DataFrame(
            [
                {
                    "timestamp": datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=i),
                    "open": o, "high": h, "low": l, "close": c, "volume": 10.0,
                }
                for i, (o, h, l, c) in enumerate(prices)
            ]
        )
        swings = detect_swings(df, left=2, right=2)
        highs = [s for s in swings if s.kind == SwingKind.HIGH]
        assert len(highs) == 1
        assert highs[0].index == 2
