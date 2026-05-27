"""Tests for Order Block detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from app.market_structure.order_blocks import detect_order_blocks
from app.schemas.domain import OBType


def _make_df(prices):
    rows = []
    start = datetime(2025, 1, 1, tzinfo=UTC)
    for i, (o, h, l, c) in enumerate(prices):
        rows.append({
            "timestamp": start + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c, "volume": 100.0,
        })
    return pd.DataFrame(rows)


class TestOrderBlocks:
    def test_bullish_ob_detected(self):
        """A bearish candle followed by a strong bullish impulse should create a bullish OB."""
        prices = [
            (100, 102, 98, 101),
            (101, 102, 99, 100),  # small bearish candle (the OB candidate)
            (100, 99, 98, 99),    # bearish candle = OB
            (99, 140, 98, 138),   # strong bullish impulse
            (138, 145, 136, 142),
        ]
        df = _make_df(prices)
        obs = detect_order_blocks(df, impulse_atr_mult=0.5)

        bullish = [ob for ob in obs if ob.ob_type == OBType.BULLISH]
        assert len(bullish) >= 1

    def test_bearish_ob_detected(self):
        """A bullish candle followed by a strong bearish impulse should create a bearish OB."""
        prices = [
            (140, 142, 138, 139),
            (139, 141, 138, 140),  # bullish candle = OB candidate
            (140, 142, 139, 141),  # bullish candle = OB
            (141, 142, 100, 102),  # strong bearish impulse
            (102, 105, 98, 100),
        ]
        df = _make_df(prices)
        obs = detect_order_blocks(df, impulse_atr_mult=0.5)

        bearish = [ob for ob in obs if ob.ob_type == OBType.BEARISH]
        assert len(bearish) >= 1

    def test_no_ob_without_impulse(self):
        """Small candles with no impulse should not produce OBs."""
        prices = [
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ]
        df = _make_df(prices)
        obs = detect_order_blocks(df, impulse_atr_mult=2.0)
        assert obs == []

    def test_ob_mitigation(self):
        """A bullish OB should be marked mitigated when price closes below it."""
        prices = [
            (100, 102, 98, 101),
            (101, 102, 99, 100),
            (100, 99, 97, 98),     # bearish candle = OB [97, 99]
            (98, 140, 97, 138),    # impulse
            (138, 140, 136, 137),
            (137, 138, 90, 92),    # price crashes below OB → mitigated
        ]
        df = _make_df(prices)
        obs = detect_order_blocks(df, impulse_atr_mult=0.5)

        bullish = [ob for ob in obs if ob.ob_type == OBType.BULLISH]
        mitigated = [ob for ob in bullish if ob.mitigated]
        assert len(mitigated) >= 1

    def test_minimum_data(self):
        """Less than 4 candles should return empty."""
        prices = [(100, 102, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)]
        df = _make_df(prices)
        assert detect_order_blocks(df) == []
