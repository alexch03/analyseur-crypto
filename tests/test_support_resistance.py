"""Tests for support/resistance level clustering."""

from __future__ import annotations

from datetime import UTC, datetime

from app.market_structure.support_resistance import detect_sr_levels
from app.market_structure.swings import detect_swings
from app.schemas.domain import SRRole


class TestSupportResistance:
    def test_range_market_clusters(self, tiny_ohlcv_range):
        """In a range market, swing highs and lows should cluster into
        at most a few SR levels."""
        swings = detect_swings(tiny_ohlcv_range, left=2, right=2)
        levels = detect_sr_levels(tiny_ohlcv_range, swings, atr_mult=0.5, min_touches=2)

        assert len(levels) >= 1, "Range market should produce at least one SR level"
        for lv in levels:
            assert lv.touches >= 2

    def test_three_close_swings_merge_into_one(self):
        """Three swings within epsilon should produce one level."""
        import pandas as pd
        from app.schemas.domain import SwingKind, SwingPoint

        swings = [
            SwingPoint(index=2, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=100.0, kind=SwingKind.LOW),
            SwingPoint(index=5, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=101.0, kind=SwingKind.LOW),
            SwingPoint(index=8, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=100.5, kind=SwingKind.LOW),
        ]
        n = 10
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)] * n,
                "open": [100.0] * n,
                "high": [110.0] * n,
                "low": [90.0] * n,
                "close": [105.0] * n,
                "volume": [10.0] * n,
            }
        )
        levels = detect_sr_levels(df, swings, atr_mult=1.0, min_touches=2)
        assert len(levels) == 1
        assert levels[0].touches == 3
        assert levels[0].role == SRRole.SUPPORT

    def test_two_distinct_clusters(self):
        """Two groups of swings far apart should produce two levels."""
        import pandas as pd
        from app.schemas.domain import SwingKind, SwingPoint

        swings = [
            SwingPoint(index=1, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=50.0, kind=SwingKind.LOW),
            SwingPoint(index=3, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=51.0, kind=SwingKind.LOW),
            SwingPoint(index=5, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=200.0, kind=SwingKind.HIGH),
            SwingPoint(index=7, timestamp=datetime(2025, 1, 1, tzinfo=UTC), price=201.0, kind=SwingKind.HIGH),
        ]
        n = 10
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2025, 1, 1, tzinfo=UTC)] * n,
                "open": [100.0] * n,
                "high": [210.0] * n,
                "low": [40.0] * n,
                "close": [105.0] * n,
                "volume": [10.0] * n,
            }
        )
        levels = detect_sr_levels(df, swings, atr_mult=0.3, min_touches=2)
        assert len(levels) == 2
        roles = {lv.role for lv in levels}
        assert SRRole.SUPPORT in roles
        assert SRRole.RESISTANCE in roles

    def test_empty_swings(self, tiny_ohlcv_bull):
        levels = detect_sr_levels(tiny_ohlcv_bull, [], atr_mult=0.5)
        assert levels == []
